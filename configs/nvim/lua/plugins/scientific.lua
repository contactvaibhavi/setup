return {
  -- Jupyter integration (without image support)
  {
    "benlubas/molten-nvim",
    version = "^1.0.0",
    build = ":UpdateRemotePlugins",
    init = function()
      vim.g.molten_output_win_max_height = 20
      vim.g.molten_auto_open_output = false
    end,
    keys = {
      { "<leader>mi", "<cmd>MoltenInit<CR>", desc = "Init Kernel" },
      { "<leader>me", "<cmd>MoltenEvaluateLine<CR>", desc = "Eval Line" },
      { "<leader>mc", "<cmd>MoltenReevaluateCell<CR>", desc = "Re-eval Cell" },
      { "<leader>mo", "<cmd>MoltenShowOutput<CR>", desc = "Show Output" },
      { "<leader>mv", "<cmd>MoltenEvaluateVisual<CR>gv", mode = "v", desc = "Eval Visual" },

      -- Calculator using bc
      {
        "<leader>qm",
        function()
          local expr = vim.fn.input("Calculate: ")
          if expr ~= "" then
            local result = vim.fn.system("bc -l", expr):gsub("^%s*(.-)%s*$", "%1")
            vim.notify(expr .. " = " .. result, vim.log.levels.INFO)
          end
        end,
        desc = "Calculator",
      },

      -- Quick plot function (opens in iTerm2)
      {
        "<leader>qp",
        function()
          local func = vim.fn.input("Function (e.g., x**2, np.sin(x)): ")
          if func ~= "" then
            -- Get the full path to python3
            local python_path = vim.fn.system("which python3"):gsub("%s+", "")

            local python_code = string.format(
              [[
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 1000)
try:
    y = %s
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title('y = %s', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.show()
except Exception as e:
    print(f"Error: {e}")
    input("Press Enter to close...")
]],
              func,
              func
            )

            local tmpfile = os.tmpname() .. ".py"
            local file = io.open(tmpfile, "w")
            file:write(python_code)
            file:close()

            -- Open in iTerm2
            local applescript = string.format(
              [[
tell application "iTerm"
    create window with default profile
    tell current session of current window
        write text "%s %s"
    end tell
end tell
]],
              python_path,
              tmpfile
            )

            local script_file = os.tmpname() .. ".scpt"
            local script = io.open(script_file, "w")
            script:write(applescript)
            script:close()

            vim.fn.system("osascript " .. script_file)
            vim.notify("Plotting: " .. func, vim.log.levels.INFO)
          end
        end,
        desc = "Quick Plot",
      },

      -- Plot activation functions comparison
      {
        "<leader>qa",
        function()
          local python_path = vim.fn.system("which python3"):gsub("%s+", "")

          local python_code = [[
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 1000)

# Activation functions
relu = np.maximum(0, x)
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
swish = x * sigmoid
elu = np.where(x > 0, x, np.exp(x) - 1)

plt.figure(figsize=(12, 8))
plt.plot(x, relu, label='ReLU', linewidth=2)
plt.plot(x, gelu, label='GELU', linewidth=2)
plt.plot(x, sigmoid, label='Sigmoid', linewidth=2)
plt.plot(x, tanh, label='Tanh', linewidth=2)
plt.plot(x, swish, label='Swish', linewidth=2)
plt.plot(x, elu, label='ELU', linewidth=2)

plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='upper left')
plt.title('Activation Functions Comparison', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.ylim(-2, 5)
plt.show()
]]

          local tmpfile = os.tmpname() .. ".py"
          local file = io.open(tmpfile, "w")
          file:write(python_code)
          file:close()

          local applescript = string.format(
            [[
tell application "iTerm"
    create window with default profile
    tell current session of current window
        write text "%s %s"
    end tell
end tell
]],
            python_path,
            tmpfile
          )

          local script_file = os.tmpname() .. ".scpt"
          local script = io.open(script_file, "w")
          script:write(applescript)
          script:close()

          vim.fn.system("osascript " .. script_file)
          vim.notify("Plotting activation functions", vim.log.levels.INFO)
        end,
        desc = "Activation Functions",
      },

      -- Custom range plot
      {
        "<leader>qr",
        function()
          local func = vim.fn.input("Function: ")
          if func == "" then
            return
          end

          local xmin = vim.fn.input("X min [-10]: ")
          xmin = xmin ~= "" and xmin or "-10"

          local xmax = vim.fn.input("X max [10]: ")
          xmax = xmax ~= "" and xmax or "10"

          local python_path = vim.fn.system("which python3"):gsub("%s+", "")

          local python_code = string.format(
            [[
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(%s, %s, 1000)
try:
    y = %s
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title('y = %s', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.show()
except Exception as e:
    print(f"Error: {e}")
    input("Press Enter to close...")
]],
            xmin,
            xmax,
            func,
            func
          )

          local tmpfile = os.tmpname() .. ".py"
          local file = io.open(tmpfile, "w")
          file:write(python_code)
          file:close()

          local applescript = string.format(
            [[
tell application "iTerm"
    create window with default profile
    tell current session of current window
        write text "%s %s"
    end tell
end tell
]],
            python_path,
            tmpfile
          )

          local script_file = os.tmpname() .. ".scpt"
          local script = io.open(script_file, "w")
          script:write(applescript)
          script:close()

          vim.fn.system("osascript " .. script_file)
          vim.notify(string.format("Plotting: %s [%s, %s]", func, xmin, xmax), vim.log.levels.INFO)
        end,
        desc = "Plot with Custom Range",
      },
    },
  },

  -- latex math preview
  {
    "jbyuki/nabla.nvim",
    keys = {
      {
        "<leader>nm",
        function()
          require("nabla").popup()
        end,
        desc = "Nabla Math Preview",
      },
    },
  },
}
