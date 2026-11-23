+++
title = 'Research'
outputs = ['html']   # no feed: papers have no pages to link to
# Papers have no pages of their own: each title links straight to the publication.
# The bundles under this folder only hold front matter and teaser images.
[[cascade]]
  [cascade.build]
    list = 'local'
    publishResources = false
    render = 'never'
  [cascade.target]
    kind = 'page'
+++
