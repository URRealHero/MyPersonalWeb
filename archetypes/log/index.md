+++
title = '{{ replace .File.ContentBaseName "-" " " | title }}'
date = {{ .Date }}
started = {{ time.Format "2006-01-02" .Date }}
status = 'active'      # active | paused | concluded
question = 'One sentence: what are you trying to find out?'
repo = ''              # link to code, if public
paper = ''             # e.g. '/research/my-paper' once this becomes one
tags = []
draft = true           # delete this line to publish
+++

{{"{{<"}} callout state {{">}}"}}
Two or three sentences you rewrite as the thread evolves: what you believe right now, and what you'd try next.
{{"{{<"}} /callout {{">}}"}}

{{"{{<"}} callout hypothesis {{">}}"}}
What do you expect to happen, and what result would change your mind?
{{"{{<"}} /callout {{">}}"}}

<!--
  Add entries ABOVE older ones, as level-2 headings that start with a date:
    ## 2026-11-20 · short title
  They become dots on the timeline, and the log index counts them automatically.
  Plots: put loss.png next to this file, then ![alt](loss.png "Caption makes it a numbered figure").
-->

## {{ time.Format "2006-01-02" .Date }} · setup

What you ran, on what data, with which config, and why.
