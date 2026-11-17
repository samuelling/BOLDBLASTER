// node build.mjs  ->  ../docs/halcyon/index.html (the game, one file; assets load from ./assets beside it)
import * as esbuild from 'esbuild';
import fs from 'fs';
const r = await esbuild.build({ entryPoints: ['src/main.js'], bundle: true, minify: true, format: 'iife', write: false, target: 'es2020', legalComments: 'none' });
const js = r.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const html = fs.readFileSync('src/index.template.html', 'utf8').replace('/*BUNDLE*/', () => js);
fs.writeFileSync('../docs/halcyon/index.html', html);
fs.mkdirSync('../docs/halcyon/harness', { recursive: true }); fs.copyFileSync('harness/bot.js', '../docs/halcyon/harness/bot.js');
console.log('built', (html.length / 1024).toFixed(0) + 'KB');
