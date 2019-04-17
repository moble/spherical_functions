---
---

# Jekyll pages with mathjax on github

In the github repo you wish to use pages for, go into settings, and
click on the page generator.  Generate the page with your favorite
layout (content doesn't matter at this point).

Now, go back to the command line, `cd` into your repo, and run `git
checkout gh-pages`.[^1]  Remove any of the garbage that git ignores (like
`build` directories, `.o` or `.pyc` files, etc.), because Jekyll will
try to serve those things as pages.

Add the following lines to a new file named `Gemfile`:

```
source 'https://rubygems.org'
gem 'github-pages'
```

Similarly, you might want to add these to your `.gitignore`:

```
_site
Gemfile.lock
```

Unless you already have Jekyll installed, you'll need to get it.  I'm
on OS X, so I use [homebrew](https://brew.sh/); package managers such
as `apt-get` should work the same.

```sh
brew install ruby
brew install libiconv # deal with bug in nokogiri installation on OS X
gem install bundler
bundle install
gem install nokogiri -- --with-iconv-dir=/usr/local/Cellar/libiconv/1.14
bundle install
```

Finally, you can start viewing your pages on
[your computer](http://localhost:4000/) with

```sh
bundle exec jekyll serve
```

You can create new pages in the top directory (or any subdirectory),
which will be get translated and be available at
https://localhost:4000/ as long as that `serve` command is running.

But for now, this will just show the `index.html` file that github created
with the page generator, and any markdown pages you use will just be
basic text pages.  You want some options and nicer templates.  Start
by doing the following:

```sh
mkdir _layouts
cp index.html _layouts/default.html
```

and edit `_layouts/default.html`.  Remove all the main matter
(probably everything inside the `<section>` tags), and replace it with

```html
{% raw %}{{ content }}{% endraw %}
```

To use mathjax, add the following lines somewhere up in the `scripts`
section:

```html
<script type="text/javascript">
  window.MathJax = {
    tex2jax: {
      inlineMath: [ ['$','$'], ],
      processEscapes: true
    }
  };
</script>
<script type="text/javascript"
        src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>
```

This allows me to enter TeX directly as usual: $e^{i\, \alpha/2}$,

\begin{equation}
  \int x^2 \mathrm{d}x = \frac{x^3}{3}.
\end{equation}

Now, add the following to `_config.yml`:

```yaml
defaults:
  -
    scope:
      path: "" # an empty string here means all files in the project
    values:
      layout: "default"

markdown:    kramdown
kramdown:
  input: GFM
  hard_wrap: false
highlighter: pygments
```

Once you restart your server, you should now be able to write simple
markdown text files in your top-level directory.  Jekyll will watch
for any changes and recreate the pages as you save them.


[^1]: Note that another (possibly better) way of doing this is to
      simply clone the `gh-pages` branch within your main repo
      directory (without making it a submodule, or anything).  Just do

          git clone -b gh-pages git@github.com:<YOUR_REPO_HERE> gh-pages

      You might also want to add `gh-pages` to `.gitignore`.
