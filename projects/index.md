---
layout: default
title: Projects
---
<h1>{{ page.title }}</h1>
<ul class="posts">
	{% for post in site.posts %}
		{% if post.published %}
			<li>
				<a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a> - 
				{% for tag in post.tags %}
					{{ tag }}
					{% if forloop.last != true %}
					• 
					{% endif %}
				{% endfor %}
				
					<br />
					<p>{{ post.blurb }}</p><br />
				<br />
			</li>
		{% endif %}
	{% endfor %}
</ul>