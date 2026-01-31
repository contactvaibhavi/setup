return {
  -- Disable default dark themes
  -- { "folke/tokyonight.nvim", enabled = false },

  { "catppuccin/nvim", enabled = false },

  -- Enable light theme
  {
    "projekt0n/github-nvim-theme",
    lazy = false,
    priority = 1000,
    config = function()
      require("github-theme").setup({
        options = {
          theme_style = "light",
        },
      })
      vim.cmd([[colorscheme github_light]])
    end,
  },
}
