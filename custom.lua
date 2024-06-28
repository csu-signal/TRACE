-- custom nvim commands for this project

local function run(dir, cmds)
    local function ret()
        local tt = require("toggleterm")

        require("toggleterm.terminal").Terminal:new {
            dir = dir,
            count = 7,
        }:open(10)

        vim.iter(cmds):map(function(x) tt.exec(x, 7) end)
    end
    return ret
end

-- build project
require("which-key").register({
    ["<leader>b"] = {
        name = "build",
        o = { run(
            "C:\\Users\\brady\\Desktop\\Camera_Calibration\\offline_processor",
            {
                "MSBuild /property:Configuration=Release",
                "set PATH=%PATH%;C:\\Users\\brady\\Desktop\\opencv\\build\\x64\\vc14\\bin",
                "build\\bin\\Release\\offline_processor.exe"
            }), "offline_processor" },
        p = { run(
            "C:\\Users\\brady\\Desktop\\Camera_Calibration",
            {
                "pip uninstall -y azure_kinect_wrapper",
                "pip install ./azure_kinect_wrapper",
                "python wrapper_test/test.py"
            }), "azure_kinect wrapper" },
        c = { function()
            require("toggleterm").toggle(7)
        end, "close" }
    }
})

-- fix comments in cpp files
vim.api.nvim_create_autocmd({ "BufEnter" }, {
    pattern = { "*.cpp", "*.hpp", "*.c", "*.h" },
    callback = function()
        vim.o.commentstring = "// %s"
        vim.o.shiftwidth = 2
    end
})
