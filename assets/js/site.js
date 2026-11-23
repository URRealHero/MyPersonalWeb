// Site-wide behaviour: BibTeX toggles, copy buttons, thumbnail videos, TOC highlight.

// BibTeX: inline expand/collapse in paper lists.
document.querySelectorAll(".bib-toggle").forEach((btn) => {
  btn.addEventListener("click", () => {
    const box = document.getElementById(btn.getAttribute("aria-controls"));
    if (!box) return;
    box.hidden = !box.hidden;
    btn.setAttribute("aria-expanded", String(!box.hidden));
  });
});

// Copy buttons: add one to every highlighted code block; handle all of them in one listener.
document.querySelectorAll(".prose .highlight").forEach((block) => {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "copy-btn";
  btn.textContent = "Copy";
  block.appendChild(btn);
});
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".copy-btn");
  if (!btn) return;
  const target = btn.dataset.copyTarget
    ? document.querySelector(btn.dataset.copyTarget)
    : btn.parentElement.querySelector("code");
  if (!target || !navigator.clipboard) return;
  navigator.clipboard.writeText(target.innerText.trim()).then(() => {
    btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = "Copy"; }, 1400);
  });
});

// Thumbnail videos (as on Barron's page): play while hovering a paper; on touch screens a tap toggles.
document.querySelectorAll(".paper").forEach((card) => {
  const thumb = card.querySelector(".one");
  const video = thumb && thumb.querySelector(".two video");
  if (!video) return;
  if (window.matchMedia("(hover: hover)").matches) {
    card.addEventListener("mouseenter", () => { video.play().catch(() => {}); });
    card.addEventListener("mouseleave", () => { video.pause(); });
  }
  let moved = false;
  thumb.addEventListener("touchstart", () => { moved = false; }, { passive: true });
  thumb.addEventListener("touchmove", () => { moved = true; }, { passive: true });
  thumb.addEventListener("touchend", () => {
    if (moved) return;
    const active = !card.classList.contains("hover-active");
    document.querySelectorAll(".paper.hover-active").forEach((other) => {
      other.classList.remove("hover-active");
      const v = other.querySelector(".two video");
      if (v) v.pause();
    });
    if (active) { card.classList.add("hover-active"); video.play().catch(() => {}); }
  }, { passive: true });
});

// TOC: highlight the section being read.
const tocLinks = Array.from(document.querySelectorAll(".toc-side a[href^='#']"));
if (tocLinks.length && "IntersectionObserver" in window) {
  const byId = new Map(tocLinks.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      tocLinks.forEach((a) => a.classList.remove("active"));
      const link = byId.get(entry.target.id);
      if (link) link.classList.add("active");
    });
  }, { rootMargin: "0px 0px -75% 0px" });
  byId.forEach((_, id) => {
    const heading = document.getElementById(id);
    if (heading) observer.observe(heading);
  });
}
