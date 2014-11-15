#! /usr/bin/env bash -l

cd $( dirname "${BASH_SOURCE[0]}" )
screen -dmS "spherical_functions-gh-pages" bundle exec jekyll serve -b /spherical_functions;
