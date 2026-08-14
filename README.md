# Parallel Bites

Notes on distributed AI systems, collective communication, parallel simulation,
and the papers behind them. Built with
[Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) and published on
GitHub Pages.

## Site

<https://ccc112233ccc.github.io>

The source lives in `ccc112233ccc/ccc112233ccc.github.io`. Pushes to `main`
are built and deployed automatically by GitHub Actions.

## Writing

Add Markdown files to `_posts` using this filename format:

```text
YYYY-MM-DD-英文短标题.md
```

The existing paper-reading template is a useful starting point. Posts may be
written in English or Chinese.

## Local preview

```shell
bundle install
bundle exec jekyll serve --livereload
```

Open <http://127.0.0.1:4000>.

## Comments

The theme is ready for Giscus. Enable GitHub Discussions, generate the values
at <https://giscus.app>, and add them to `comments.giscus` in `_config.yml`.
