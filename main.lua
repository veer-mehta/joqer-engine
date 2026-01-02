-- Load dkjson
local json = dofile(SMODS.current_mod.path .. "/dkjson.lua")
local DECISION_COOLDOWN = 0.3
local auto_timer = 0
local phase = "idle"
local pending_decision = nil


-- Call Python LLM
local function call_llm(state)
    love.filesystem.write("state.json", json.encode(state, { indent = true }))

	local cwd = love.filesystem.getWorkingDirectory()

    os.execute(cwd .. [[/Mods/AutoPlayer/apenv/Scripts/activate.bat && python Mods/AutoPlayer/decide.py]])

    if not love.filesystem.getInfo("decision.json") then
        return nil
    end

    local raw = love.filesystem.read("decision.json")
    local ok, decision = pcall(json.decode, raw)
    if not ok then return nil end

    if type(decision) ~= "table" then return nil end
    if decision.action ~= "play" and decision.action ~= "discard" then return nil end
    if type(decision.card_indexes) ~= "table" then return nil end

    return decision
end


-- Capture game state
local function get_game_state()
    if not G or not G.GAME or not G.hand then return nil end

    local state = {
        state = G.STATE,
        round = G.GAME.round,
        stake = G.GAME.stake,
        chips = G.GAME.chips,
        mult = G.GAME.mult,
        dollars = G.GAME.dollars,
        hands_played = G.GAME.hands_played,
        unused_discards = G.GAME.unused_discards,
        poker_hands = {},
        hand = {},
        jokers = {}
    }

    if G.GAME.blind then
        state.blind = {
            name = G.GAME.blind.name,
            chips_required = G.GAME.blind.chips,
            is_boss = G.GAME.blind.boss
        }
    end

    if G.GAME.hands then
        for name, hand in pairs(G.GAME.hands) do
            state.poker_hands[name] = {
                level = hand.level or 1,
                times_played = hand.played or 0,
                chips = hand.chips or 0,
                mult = hand.mult or 0
            }
        end
    end

    for _, card in ipairs(G.hand.cards or {}) do
        table.insert(state.hand, {
            rank = card.base and card.base.id,
            value = card.base and card.base.value,
            suit = card.base and card.base.suit
        })
    end

    if G.jokers and G.jokers.cards then
        for _, joker in ipairs(G.jokers.cards) do
            table.insert(state.jokers, {
                name = joker.ability and joker.ability.name
            })
        end
    end

    return state
end


local game_update_ref = Game.update

function Game.update(self, dt)
    game_update_ref(self, dt)

    if G.STATE ~= G.STATES.SELECTING_HAND then
        auto_timer = 0
        phase = "idle"
        pending_decision = nil
        return
    end

    auto_timer = auto_timer + dt
    if auto_timer < DECISION_COOLDOWN then return end


    -- Request decision
    if phase == "idle" then
        local state = get_game_state()
        if not state then return end

        pending_decision = call_llm(state)
        if not pending_decision then return end

        phase = "select"
        auto_timer = 0
        return
    end

    -- Select cards
    if phase == "select"
	and pending_decision then
        for _, idx in ipairs(pending_decision.card_indexes) do
            if idx >= 0 then
                local card = G.hand.cards[idx + 1]
                if card then card:click() end
            end
        end

        phase = "execute"
        return
    end

    -- Execute action
    if phase == "execute"
	and pending_decision then
        if pending_decision.action == "play" then
            local btn = G.buttons:get_UIE_by_ID("play_button")
            if btn and btn.config and btn.config.button then
                G.FUNCS[btn.config.button](btn)
            end

        elseif pending_decision.action == "discard" then
            local btn = G.buttons:get_UIE_by_ID("discard_button")
            if btn and btn.config and btn.config.button then
                G.FUNCS[btn.config.button](btn)
            end
        end

        phase = "cooldown"
        auto_timer = 0
        return
    end

    -- Cooldown
    if phase == "cooldown" then
        auto_timer = auto_timer + dt
        if auto_timer > 0.3 then
            phase = "idle"
            pending_decision = nil
            auto_timer = 0
        end
    end
end
