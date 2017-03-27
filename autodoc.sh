#!/usr/bin/env bash

# build the docs
cd docs
make clean
make html
cd ..

# commit and push
git add -A
git commit -m "building and pushing docs"
git push origin master

cd ../docs/html