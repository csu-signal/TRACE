-- custom nvim commands for this project

-- build project
vim.keymap.set("n", "<leader>b", function()
    local tt = require("toggleterm")

    require("toggleterm.terminal").Terminal:new {
        dir = "C:\\Users" ..
            "\\brady\\Desktop\\Camera_Calibration" ..
            "\\offline_processor",
        count = 7,
    }:open(10)

    tt.exec("MSBuild /property:Configuration=Release", 7)
    tt.exec("set PATH=%PATH%;C:\\Users\\brady\\Desktop\\opencv\\build\\x64\\vc14\\bin", 7)
    tt.exec("build\\bin\\Release\\offline_processor.exe", 7)
end)

vim.keymap.set("n", "<leader>B", function()
    require("toggleterm").toggle(7)
end)
