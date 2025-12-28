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
      -- Popup calculator using bc
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
