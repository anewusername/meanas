#!/bin/bash

set -Eeuo pipefail

cd ~/projects/meanas

# Approach 1: pdf to html?
#pdoc3 --pdf --force --template-dir pdoc_templates -o doc . | \
#    pandoc --metadata=title:"meanas" --toc --toc-depth=4 --from=markdown+abbreviations --to=html --output=doc.html --gladtex -s -

# Approach 2: pdf to html with gladtex
rm -r _doc_mathimg
pdoc --pdf --force --template-dir pdoc_templates -o doc . > doc.md
pandoc --metadata=title:"meanas" --from=markdown+abbreviations --to=html --output=doc.htex --gladtex -s --css pdoc_templates/pdoc.css doc.md
gladtex -a -n -d _doc_mathimg -c white doc.htex

# Approach 3: html with gladtex
#pdoc3 --html --force --template-dir pdoc_templates -o doc .
#find doc -iname '*.html' -exec gladtex -a -n -d _mathimg -c white {} \;
