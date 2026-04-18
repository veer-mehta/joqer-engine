local ai_request = love.thread.getChannel("ai_request")
local ai_response = love.thread.getChannel("ai_response")

local curr_mod_path = "Mods/JoQerEngine/"

local round_state_json_path = curr_mod_path .. "round_state.json"
local decision_json_path = curr_mod_path .. "decision.json"

while true do
    local msg = ai_request:demand()

    if msg == "init" then
        print("THREAD <- init")
        os.execute("conda activate apenv && python " .. curr_mod_path .. "/decide.py")

        print("THREAD -> fini")
        ai_response:push("fini")
    elseif msg == "quit" then
        break
    end
end
