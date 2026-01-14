---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Home
---

<div class="hero">
  <h1>Hi, I'm Pietro.</h1>
  <p>
    I am an AI engineer specialised in <b> Deep Learning </b> and <b>NLP</b>.
    I am focused on building AI systems for real-world impact.
  </p>
  <div style="margin-top: 20px;">
    <a href="/about/" class="btn">About Me</a>
    <a href="https://github.com/" target="_blank" class="btn" style="background:#333; margin-left:10px;">GitHub</a>
  </div>
</div>

<hr style="border: 0; border-top: 1px solid #eee; margin: 50px 0;">

<h2 style="text-align: center; border: none;">Selected Projects</h2>

<div class="projects-grid">
  {% assign featured_projects = site.pages | where: "featured", true %}
  {% for featured_project in featured_projects %}
  <div class="project-card">
    <img src="{{featured_project.image}}" class="card-img" alt="Project Image">
    <div class="card-content">
      <h3 class="c  ard-title">{{featured_project.title}}</h3>
      <p class="card-desc">
        {{featured_project.description}}
      </p>
      <div class="card-tags">
          {% for tool in featured_project.tools %}
            <span class="skill-tag">{{tool}}</span>
          {% endfor %}
      </div>
      <a href="{{featured_project.permalink}}" class="card-link">Read Case Study →</a>
    </div>
  </div>
  {% endfor %}
</div>
