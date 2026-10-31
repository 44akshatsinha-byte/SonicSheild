(() => {
  const layer = document.getElementById("noise");
  if (!layer) return;
  const size = 256;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(size, size);
  const data = img.data;
  for (let i = 0; i < data.length; i += 4) {
    const v = 170 + Math.random() * 80;
    data[i] = v;
    data[i + 1] = v;
    data[i + 2] = v;
    data[i + 3] = 40 + Math.random() * 50;
  }
  ctx.putImageData(img, 0, 0);
  layer.style.backgroundImage = `url(${c.toDataURL("image/png")})`;
  layer.style.backgroundRepeat = "repeat";
  layer.style.backgroundSize = "380px 380px";
})();
