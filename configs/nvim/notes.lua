return {
  -- Obsidian integration for notes
  {
    "epwalsh/obsidian.nvim",
    version = "*",
    lazy = true,
    ft = "markdown",
    dependencies = {
      "nvim-lua/plenary.nvim",
      "nvim-telescope/telescope.nvim",
    },
    opts = {
      workspaces = {
        {
          name = "notes",
          path = "~/notes",
        },
        {
          name = "journal",
          path = "~/journal",
        },
      },
      daily_notes = {
        folder = "daily",
        date_format = "%Y-%m-%d",
      },
    },
    keys = {
      { "<leader>on", "<cmd>ObsidianNew<cr>", desc = "New Note" },
      { "<leader>oo", "<cmd>ObsidianSearch<cr>", desc = "Search Notes" },
      { "<leader>ot", "<cmd>ObsidianToday<cr>", desc = "Today's Note" },
    },
  },
  {
    "toppair/peek.nvim",
    build = "deno task --quiet build:fast",
    ft = { "markdown" },
    config = function()
      require("peek").setup()
      vim.api.nvim_create_user_command("PeekOpen", require("peek").open, {})
      vim.api.nvim_create_user_command("PeekClose", require("peek").close, {})
    end,
    keys = {
      { "<leader>mp", "<cmd>PeekOpen<cr>", desc = "Peek Markdown" },
    },
  },
}
