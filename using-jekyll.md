---
title: Blogging Like a Hacker
---

* In the github repo, go into settings, and click on the page generator
* Generate the page with your favorite layout (content doesn't matter)
* On the command line, `cd` into your repo and run
  * git checkout gh-pages
* Be sure to remove any of the garbage that git ignores (like `build`
  directories, `.o` or `.pyc` files, etc.)
* Add the following lines to a new file named `Gemfile`:
  ```
  source 'https://rubygems.org'
  gem 'github-pages'
  ```
* `brew install ruby`
* `brew install libiconv` (to deal with bug in next line)
* `gem install bundler`
* `bundle install`
* `gem install nokogiri -- --with-iconv-dir=/usr/local/Cellar/libiconv/1.14`
* `bundle install`
* `bundle exec jekyll serve`

For now, this will just show the `index.html` file that github created
with the page generator.  You can create new pages in the top
directory (or any subdirectory)
