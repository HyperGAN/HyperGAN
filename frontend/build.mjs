import { build } from "esbuild";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("./", import.meta.url));
const result = await build({
  absWorkingDir: root,
  entryPoints: ["src/app.js"],
  bundle: true,
  format: "esm",
  target: ["es2022"],
  minify: true,
  legalComments: "none",
  write: false,
});
const target = new URL("../src/hypergan/web_assets/app.js", import.meta.url);
let notices = "HyperGAN local viewer bundled third-party licenses\n\n";
for (const name of ["echarts", "zrender", "tslib"]) {
  const pkg = JSON.parse(
    await readFile(
      new URL(`node_modules/${name}/package.json`, import.meta.url),
      "utf8",
    ),
  );
  notices += `\n=== ${name} ${pkg.version} ===\n`;
  for (const filename of name === "tslib"
    ? ["CopyrightNotice.txt", "LICENSE.txt"]
    : ["LICENSE", "NOTICE"]) {
    try {
      notices += await readFile(
        new URL(`node_modules/${name}/${filename}`, import.meta.url),
        "utf8",
      );
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
}
const outputs = [
  [target, result.outputFiles[0].contents],
  [
    new URL(
      "../src/hypergan/web_assets/THIRD_PARTY_LICENSES.txt",
      import.meta.url,
    ),
    Buffer.from(notices.replace(/\r\n/g, "\n")),
  ],
];
for (const [path, bytes] of outputs) {
  if (process.argv.includes("--check")) {
    const actual = await readFile(path);
    if (!actual.equals(Buffer.from(bytes)))
      throw new Error(`Bundled output differs: ${path}`);
  } else await writeFile(path, bytes);
}
console.log(
  `Viewer bundle ${result.outputFiles[0].contents.length} bytes (${process.argv.includes("--check") ? "verified" : "built"})`,
);
