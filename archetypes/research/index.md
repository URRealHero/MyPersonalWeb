+++
title = '{{ replace .File.ContentBaseName "-" " " | title }}'
short = ''             # short name, used in log links
date = {{ .Date }}     # arXiv v1 date (sets the order)
year = {{ now.Year }}  # venue year
venue = ''             # e.g. 'CVPR', 'ICLR'
badge = ''             # e.g. 'Oral', 'Spotlight' (leave empty for none)
selected = true        # show on the homepage
highlight = false      # yellow row, like Barron's highlighted papers
authors = ['Shu Pu']   # suffix '*' for equal contribution
# The paper's own question or a brief intro, taken from the paper.
# A list shows as Q1, Q2, …: hook = ['First question?', 'Second question?']
hook = ''
teaser_video = ''      # optional mp4 URL; plays on hover
bibtex = '''
'''
draft = true

[links]
  paper = ''           # the title and teaser link here
  code = ''
  project = ''
+++
<!-- Papers have no page of their own; only the front matter above is used.
     Put teaser.png (or .jpg) next to this file. -->
