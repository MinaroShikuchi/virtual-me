import reflex as rx

config = rx.Config(
    app_name="virtual_me",
    env_file=".env",
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
)
