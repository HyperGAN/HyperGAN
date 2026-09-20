import { build, context } from "esbuild";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("./", import.meta.url));
const check = process.argv.includes("--check");
const watch = process.argv.includes("--watch");
if (check && watch) throw new Error("--check and --watch are exclusive");
const options = {
  absWorkingDir: root,
  entryPoints: ["src/app.js"],
  bundle: true,
  format: "esm",
  target: ["es2022"],
  minify: true,
  legalComments: "none",
  write: false,
};
const target = new URL("../src/hypergan/web_assets/app.js", import.meta.url);
const licenses = new URL(
  "../src/hypergan/web_assets/THIRD_PARTY_LICENSES.txt",
  import.meta.url,
);

async function notices() {
  let text = "HyperGAN local viewer bundled third-party licenses\n\n";
  for (const name of ["echarts", "zrender", "tslib"]) {
    const pkg = JSON.parse(
      await readFile(
        new URL(`node_modules/${name}/package.json`, import.meta.url),
        "utf8",
      ),
    );
    text += `\n=== ${name} ${pkg.version} ===\n`;
    for (const filename of name === "tslib"
      ? ["CopyrightNotice.txt", "LICENSE.txt"]
      : ["LICENSE", "NOTICE"]) {
      try {
        text += await readFile(
          new URL(`node_modules/${name}/${filename}`, import.meta.url),
          "utf8",
        );
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    }
  }
  return Buffer.from(text.replace(/\r\n/g, "\n"));
}

// One emitter for every mode, so a watched rebuild writes the same bytes as
// `npm run build` and never diverges from what `--check` verifies.
async function emit(result, label) {
  const outputs = [
    [target, result.outputFiles[0].contents],
    [licenses, await notices()],
  ];
  for (const [path, bytes] of outputs) {
    if (check) {
      const actual = await readFile(path);
      if (!actual.equals(Buffer.from(bytes)))
        throw new Error(`Bundled output differs: ${path}`);
    } else await writeFile(path, bytes);
  }
  console.log(
    `Viewer bundle ${result.outputFiles[0].contents.length} bytes (${label})`,
  );
}

if (watch) {
  const ctx = await context({
    ...options,
    plugins: [
      {
        name: "hypergan-emit",
        setup(build) {
          build.onEnd(async (result) => {
            if (result.errors.length) return;
            await emit(result, "rebuilt");
          });
        },
      },
    ],
  });
  await ctx.watch();
  console.log(
    "Watching frontend/src; refresh the viewer in the browser after each rebuild.",
  );
} else {
  await emit(await build(options), check ? "verified" : "built");
}
