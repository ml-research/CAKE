#!/usr/bin/env sh
cp ~/.netrc ./
cp ~/.pdbrc ./
docker build -t cake .
