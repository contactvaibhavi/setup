return {
  -- Python support
  {
    "python-mode/python-mode",
    ft = "python",
    branch = "develop",
    config = function()
      vim.g.pymode_python = "python3"
      vim.g.pymode_lint = 0
      vim.g.pymode_rope = 0
      vim.g.pymode_options_colorcolumn = 0
    end,
  },

  -- Better Python indentation
  {
    "Vimjas/vim-python-pep8-indent",
    ft = "python",
  },

  -- Python docstring generator
  {
    "danymat/neogen",
    dependencies = "nvim-treesitter/nvim-treesitter",
    keys = {
      {
        "<leader>pd",
        function()
          require("neogen").generate()
        end,
        ft = "python",
        desc = "Generate Docstring",
      },
    },
    config = function()
      require("neogen").setup({
        enabled = true,
        languages = {
          python = {
            template = {
              annotation_convention = "google",
            },
          },
        },
      })
    end,
  },

  -- Python REPL
  {
    "jpalardy/vim-slime",
    ft = "python",
    keys = {
      { "<leader>ps", "<Plug>SlimeSendCell", ft = "python", desc = "Send Cell to REPL" },
    },
    config = function()
      vim.g.slime_target = "tmux"
      vim.g.slime_python_ipython = 1
    end,
  },

  -- Debug Adapter Protocol (DAP) for Python debugging
  {
    "mfussenegger/nvim-dap",
    dependencies = {
      "mfussenegger/nvim-dap-python",
      "rcarriga/nvim-dap-ui",
      "theHamsta/nvim-dap-virtual-text",
      "nvim-neotest/nvim-nio",
    },
    ft = "python",
    keys = {
      { "<leader>db", "<cmd>DapToggleBreakpoint<CR>", desc = "Toggle Breakpoint" },
      { "<leader>dc", "<cmd>DapContinue<CR>", desc = "Continue" },
      { "<leader>di", "<cmd>DapStepInto<CR>", desc = "Step Into" },
      { "<leader>do", "<cmd>DapStepOver<CR>", desc = "Step Over" },
      { "<leader>du", "<cmd>lua require('dapui').toggle()<CR>", desc = "Toggle DAP UI" },
    },
    config = function()
      local dap = require("dap")
      local dapui = require("dapui")

      require("dap-python").setup("python3")
      require("nvim-dap-virtual-text").setup()
      dapui.setup()

      -- Auto open/close UI
      dap.listeners.after.event_initialized["dapui_config"] = function()
        dapui.open()
      end
      dap.listeners.before.event_terminated["dapui_config"] = function()
        dapui.close()
      end
    end,
  },

  -- Python virtual environment selector
  {
    "linux-cultist/venv-selector.nvim",
    dependencies = {
      "neovim/nvim-lspconfig",
      "nvim-telescope/telescope.nvim",
      "mfussenegger/nvim-dap-python",
    },
    ft = "python",
    keys = {
      { "<leader>pv", "<cmd>VenvSelect<CR>", desc = "Select Python VEnv" },
    },
    opts = {
      name = { "venv", ".venv", "env", ".env" },
    },
  },

  -- pytest integration
  {
    "nvim-neotest/neotest",
    dependencies = {
      "nvim-lua/plenary.nvim",
      "nvim-treesitter/nvim-treesitter",
      "nvim-neotest/neotest-python",
      "nvim-neotest/nvim-nio",
    },
    ft = "python",
    keys = {
      { "<leader>tt", "<cmd>lua require('neotest').run.run()<CR>", desc = "Run Nearest Test" },
      { "<leader>tf", "<cmd>lua require('neotest').run.run(vim.fn.expand('%'))<CR>", desc = "Run File Tests" },
      { "<leader>td", "<cmd>lua require('neotest').run.run({strategy = 'dap'})<CR>", desc = "Debug Test" },
      { "<leader>ts", "<cmd>lua require('neotest').summary.toggle()<CR>", desc = "Toggle Test Summary" },
      { "<leader>to", "<cmd>lua require('neotest').output.open({ enter = true })<CR>", desc = "Show Test Output" },
    },
    config = function()
      require("neotest").setup({
        adapters = {
          require("neotest-python")({
            dap = { justMyCode = false },
            runner = "pytest",
          }),
        },
      })
    end,
  },

  -- Python refactoring tools
  {
    "ThePrimeagen/refactoring.nvim",
    dependencies = {
      "nvim-lua/plenary.nvim",
      "nvim-treesitter/nvim-treesitter",
    },
    ft = "python",
    keys = {
      {
        "<leader>re",
        "<cmd>lua require('refactoring').refactor('Extract Function')<CR>",
        mode = "v",
        desc = "Extract Function",
      },
      {
        "<leader>rv",
        "<cmd>lua require('refactoring').refactor('Extract Variable')<CR>",
        mode = "v",
        desc = "Extract Variable",
      },
      {
        "<leader>ri",
        "<cmd>lua require('refactoring').refactor('Inline Variable')<CR>",
        desc = "Inline Variable",
      },
    },
    config = function()
      require("refactoring").setup()
    end,
  },

  -- f-string conversion helper
  {
    "roobert/f-string-toggle.nvim",
    ft = "python",
    keys = {
      {
        "<leader>pf",
        function()
          require("f-string-toggle").toggle()
        end,
        desc = "Toggle f-string",
      },
    },
    config = function()
      require("f-string-toggle").setup({
        key_binding = "<leader>pf",
        key_binding_desc = "Toggle f-string",
      })
    end,
  },

  -- Import sorting and formatting
  {
    "stsewd/isort.nvim",
    ft = "python",
    build = ":UpdateRemotePlugins",
    keys = {
      { "<leader>pi", "<cmd>Isort<CR>", desc = "Sort Imports" },
    },
  },

  -- Python semantic highlighting
  {
    "numirias/semshi",
    ft = "python",
    build = ":UpdateRemotePlugins",
  },

  -- Jupyter integration (already included, keeping for completeness)
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

      -- Calculator and plotting (keeping existing)
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

      -- ... (keep your existing plotting functions)
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
