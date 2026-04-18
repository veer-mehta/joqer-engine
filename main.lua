local curr_mod_path = "Mods/JoQerEngine/"
local json = dofile(curr_mod_path .. "dkjson.lua")

local select_timer = 0
local sgu = Game.update

local decision_thread = love.thread.newThread(curr_mod_path .. "decision_handler.lua")
local ai_request = love.thread.getChannel("ai_request")
local ai_response = love.thread.getChannel("ai_response")

local round_state_json_path = curr_mod_path .. "round_state.json"
local decision_json_path = curr_mod_path .. "decision.json"

local decision_busy = false
local click_button = nil
local pending_click = nil

decision_thread:start()


local function get_round_state()
    if not G or not G.GAME or not G.hand then return nil end

    local cr = G.GAME.current_round  -- shorthand

    local state = {
        hands_played = G.GAME.hands_played,
        hands_left = G.GAME.current_round.hands_left,
        unused_discards = G.GAME.current_round.discards_left,
        poker_hands = {},
        hand = {},
        jokers = {},
        timestamp = os.date("%Y-%m-%d %H:%M:%S")
    }


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
            suit = card.base and card.base.suit
        })
    end

    return state
end



local function start_decision(state)
    if decision_busy then return end

    love.filesystem.write(
        round_state_json_path,
        json.encode(state, { indent = true })
    )

    print("MAIN -> init")
    ai_request:push("init")
    decision_busy = true
end



local function disselect_cards()
    for _, c in ipairs(G.hand.highlighted or {}) do
        c:click()
    end
end



local function select_cards(decision)
    disselect_cards()

    -- Select cards
    for _, idx in ipairs(decision.card_indexes or {}) do
        if idx >= 0 then
            local card = G.hand.cards[idx + 1]
            if card then
                card:click()
            end
        end
    end

    -- Delay click by 1 frame
    pending_click = decision.action
end



function Game.update(self, dt)
    sgu(self, dt)

    select_timer = select_timer + dt

    if G.STATE ~= G.STATES.SELECTING_HAND then
        select_timer = 0
        return
    end


    -- Step 1: apply delayed click
    if pending_click then
        click_button = pending_click
        pending_click = nil
        return
    end


    -- Step 2: click button
    if click_button then
        local id = click_button .. "_button"

        print("clicking:", id)

        local btn = G.buttons:get_UIE_by_ID(id)
        if btn and btn.config and btn.config.button then
            G.FUNCS[btn.config.button](btn)
        else
            print("BUTTON NOT FOUND: ", id)
        end

        click_button = nil
        return
    end


    -- Step 3: receive decision
    if decision_busy then
        local message = ai_response:pop()

        if message == "fini" then
            print("MAIN <- fini")
            decision_busy = false

            if love.filesystem.getInfo(decision_json_path) then
                local raw = love.filesystem.read(decision_json_path)
                local ok, decision = pcall(json.decode, raw)

                if ok and decision and not decision.error then
                    print("ACTION:", decision.action)
                    select_cards(decision)
                else
                    print("INVALID DECISION JSON")
                end
            else
                print("decision.json missing")
            end

            select_timer = 0
        end
    end


    -- IMPORTANT FIX: don't block when we still need to click
    if G.hand and G.hand.highlighted and #G.hand.highlighted > 0 and not click_button and not pending_click then
        select_timer = 0
        return
    end


    -- Step 4: trigger new decision
    if select_timer > 0.125 then
        local state = get_round_state()
        if state then
            start_decision(state)
        end
    end
end