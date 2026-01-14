---
layout: default
title: Projects
permalink: /projects/
---

<div class="hero" style="padding: 40px 0; text-align: left;">
  <h1>My Project Archive</h1>
  <p>A complete list of my experiments in AI, CV, and NLP.</p>
</div>

<div class="projects-grid">
  
  {% assign projects = site.pages | where: "layout", "project" %}

  {% for project in projects %}
    
    <div class="project-card">
      {% if project.image %}
        <img src="{{ project.image }}" class="card-img" alt="{{ project.title }}">
      {% else %}
        <img src="https://via.placeholder.com/600x400?text={{ project.title | url_encode }}" class="card-img" alt="Placeholder">
      {% endif %}

      <div class="card-content">
        <h3 class="card-title">{{ project.title }}</h3>
        <p class="card-desc">{{ project.description }}</p>
        
        <div class="card-tags">
          {% for tool in project.tools limit:3 %}
            <span class="skill-tag">{{ tool }}</span>
          {% endfor %}
        </div>

        <a href="{{ project.url }}" class="card-link">View Project →</a>
      </div>
    </div>

  {% endfor %}

</div>
