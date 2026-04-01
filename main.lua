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


-- Single unified state function.
-- This exact object is written to state.json (read by decide.py AND decide_bc.py)
-- and also logged to dataset.jsonl — so training data and live inference
-- always see an identical state shape.
local function get_state()
    if not G or not G.GAME or not G.hand then return nil end

    local cr = G.GAME.current_round  -- shorthand

    local state = {
        hand            = {},
        deck_remaining  = {},
        unused_discards = cr and cr.discards_left or 0,
        hands_left      = cr and cr.hands_left    or 0,
        chips_required  = G.GAME.blind and G.GAME.blind.chips or 0,
        chips_scored    = G.GAME.chips or 0,
        poker_hands     = {}
    }

    -- Full hand — no upper limit (Juggler etc. can expand beyond 8)
    for _, card in ipairs(G.hand.cards or {}) do
        table.insert(state.hand, {
            rank = card.base and card.base.id,
            suit = card.base and card.base.suit
        })
    end

    -- Full remaining deck
    for _, card in ipairs((G.deck and G.deck.cards) or {}) do
        table.insert(state.deck_remaining, {
            rank = card.base and card.base.id,
            suit = card.base and card.base.suit
        })
    end

    -- Poker hands: chips, mult, level only
    if G.GAME.hands then
        for name, hand in pairs(G.GAME.hands) do
            state.poker_hands[name] = {
                chips = hand.chips or 0,
                mult  = hand.mult  or 0,
                level = hand.level or 1
            }
        end
    end

    return state
end

-- Append one JSONL entry to dataset.jsonl (silent — no print, no UI effect)
local function log_decision(state, decision)
    if not state or not decision then return end
    state.decision = {
        action       = decision.action,
        card_indexes = decision.card_indexes
    }
    love.filesystem.append("dataset.jsonl", json.encode(state) .. "\n")
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
        local state = get_state()  -- single unified state
        if not state then return end

        pending_decision = call_llm(state)
        if not pending_decision then return end

        log_decision(state, pending_decision)  -- silent BC dataset logging

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
