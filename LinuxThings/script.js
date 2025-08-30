<script>
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("pre > code").forEach((codeBlock) => {
    const button = document.createElement("button");
    button.textContent = "Copy";
    button.className = "copy-btn";

    button.addEventListener("click", () => {
      navigator.clipboard.writeText(codeBlock.innerText).then(() => {
        button.textContent = "Copied!";
        setTimeout(() => (button.textContent = "Copy"), 2000);
      });
    });

    const pre = codeBlock.parentNode;
    pre.style.position = "relative";
    button.style.position = "absolute";
    button.style.top = "5px";
    button.style.right = "5px";
    pre.appendChild(button);
  });
});
</script>

<style>
.copy-btn {
  background: #444;
  color: #fff;
  border: none;
  padding: 2px 6px;
  font-size: 12px;
  border-radius: 4px;
  cursor: pointer;
}
.copy-btn:hover {
  background: #666;
}
</style>

<script>
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("input[type=checkbox]").forEach((box) => {
    box.disabled = false; // enable clicking
  });
});
</script>

<script>
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("input[type=checkbox]").forEach((box, i) => {
    const key = "checkbox-" + i;
    // Load saved state
    if (localStorage.getItem(key) === "true") box.checked = true;

    box.disabled = false;
    box.addEventListener("change", () => {
      localStorage.setItem(key, box.checked);
    });
  });
});
</script>


