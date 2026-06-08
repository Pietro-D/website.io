---
layout: default
title: Home
---

<div class="hero-container">
  <div class="hero-image-side">
    <img src="\assets\images\profile.jpeg" alt="Pietro" class="hero-profile-img">
  </div>

  <div class="hero-content-main">
  <h1 class="hero-name">Pietro Di Nicco</h1>
  
  <p class="hero-subtitle">
    <strong>Deep Learning</strong> & <strong>NLP</strong> Specialist
  </p>

  <div class="hero-education">
    <div class="edu-item">
      <span class="edu-icon">◈</span>
      <span>MsC in Artificial Intelligence & Deep Learning</span>
    </div>
    <div class="edu-item">
      <span class="edu-icon">◈</span>
      <span>BsC in Computer Engineering</span>
    </div>
  </div>

  <div class="hero-cta">
    <a href="/about/" class="btn btn-primary">Full Background</a>
    <a href="/projects/" class="btn btn-outline">View Projects</a>
  </div>
</div>

  <div class="hero-contact-side">
    <span class="contact-title">Get in Touch</span>
    <div class="social-links">
      <a href="mailto:diniccopietro@gmail.com" class="social-item">
        <span class="social-icon">✉ diniccopietro@gmail.com</span>
      </a>
      <a href="https://www.linkedin.com/{{linkedin_username}}" target="_blank" class="social-item">
        <span class="social-icon">Linkedin</span> 
      </a>
      <a href="https://github.com/{{github_username}}" target="_blank" class="social-item">
        <span class="social-icon">Github</span> 
      </a>
    </div>
  </div>
</div>


<hr class="section-divider">

<div class="featured-header">
  <h2>Selected Projects</h2>
  <p>A complete list of my AI, computer vision, and NLP experiments from my MSc studies.</p>
</div>

<div class="projects-grid">
  {% assign featured_projects = site.pages | where: "featured", true %}
  {% for project in featured_projects %}
  <div class="project-card">
    <div class="card-image-wrapper">
      <img src="{{project.image}}" class="card-img" alt="{{project.title}}">
    </div>
    <div class="card-content">
      <h3 class="card-title">{{project.title}}</h3>
      <p class="card-desc">{{project.description}}</p>
      <div class="card-tags">
        {% for tool in project.tools %}
        <span class="skill-tag">{{tool}}</span>
        {% endfor %}
      </div>
      <a href="{{project.permalink}}" class="card-link">Explore Case Study →</a>
    </div>
  </div>
  {% endfor %}
</div>
