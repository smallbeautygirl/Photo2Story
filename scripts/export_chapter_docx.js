#!/usr/bin/env node
/**
 * Export a thesis chapter written in Markdown to a Word document for the advisor.
 *
 * Handles the Markdown subset the chapters use: # / ## headings, paragraphs with
 * **bold** and `code`, > block quotes (rendered as indented notes), pipe tables,
 * and ![caption](figures/x.svg) images (the .png next to the .svg is embedded,
 * since Word's SVG support varies). `[CITATION NEEDED: ...]` markers are kept and
 * highlighted so they are easy to spot.
 *
 * Usage:
 *   NODE_PATH=<dir containing node_modules with docx> \
 *   node scripts/export_chapter_docx.js docs/thesis/chapter3_methodology.md docs/thesis/exports/<name>.docx
 */
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, HeadingLevel, ImageRun, Packer, Paragraph,
  ShadingType, Table, TableCell, TableRow, TextRun, WidthType, HighlightColor,
} = require("docx");

const [, , inputPath, outputPath] = process.argv;
if (!inputPath || !outputPath) {
  console.error("usage: export_chapter_docx.js <chapter.md> <out.docx>");
  process.exit(1);
}

const CJK_FONT = "新細明體"; // PMingLiU: the usual body face for Chinese theses
const CJK_HEADING_FONT = "微軟正黑體";
const LATIN_FONT = "Times New Roman";
const BODY_PT = 12;
const PAGE_W_DXA = 11906; // A4
const PAGE_H_DXA = 16838;
const MARGIN_DXA = 1440; // 2.54 cm
const TEXT_W_DXA = PAGE_W_DXA - 2 * MARGIN_DXA;
const CITATION_RE = /\[CITATION NEEDED:[^\]]*\]/g;

function runs(text, base = {}) {
  // Split on **bold**, `code` and [CITATION NEEDED: ...] while keeping order.
  const out = [];
  const token = /(\*\*[^*]+\*\*|`[^`]+`|\[CITATION NEEDED:[^\]]*\])/g;
  let last = 0;
  for (const m of text.matchAll(token)) {
    if (m.index > last) out.push(new TextRun({ ...base, text: text.slice(last, m.index) }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ ...base, text: t.slice(2, -2), bold: true }));
    else if (t.startsWith("`")) {
      const inner = t.slice(1, -1);
      // The chapters write citation placeholders inside backticks; keep them highlighted, not monospace.
      if (inner.startsWith("[CITATION NEEDED")) out.push(new TextRun({ ...base, text: inner, highlight: "yellow" }));
      else out.push(new TextRun({ ...base, text: inner, font: "Consolas" }));
    }
    else out.push(new TextRun({ ...base, text: t, highlight: "yellow" }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ ...base, text: text.slice(last) }));
  return out;
}

function paragraph(text, opts = {}) {
  return new Paragraph({
    children: runs(text, opts.run),
    spacing: { line: 360, after: 120 }, // 1.5 lines
    alignment: AlignmentType.JUSTIFIED,
    indent: opts.indent,
    border: opts.border,
    shading: opts.shading,
    ...(opts.paragraph || {}),
  });
}

function heading(text, level) {
  return new Paragraph({
    text,
    heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
    spacing: { before: level === 1 ? 480 : 360, after: 200 },
  });
}

function note(text) {
  return paragraph(text, {
    run: { italics: false, color: "444444", size: 20 },
    indent: { left: 360, right: 360 },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: "999999", space: 8 } },
  });
}

function image(mdPath, altText, baseDir) {
  const png = path.resolve(baseDir, mdPath.replace(/\.svg$/, ".png"));
  const data = fs.readFileSync(png);
  // Scale to text width; the figure is 3420x1200 px.
  const widthPx = 620;
  const heightPx = Math.round(widthPx * 1200 / 3420);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120 },
    keepNext: true, // keep the caption on the same page as the figure
    children: [new ImageRun({ type: "png", data, transformation: { width: widthPx, height: heightPx }, altText: { title: altText, description: altText, name: altText } })],
  });
}

function table(lines) {
  const rows = lines.map((l) => l.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  const body = rows.filter((_, i) => i !== 1); // drop the |---| separator
  const cols = body[0].length;
  const weights = cols === 4 ? [0.17, 0.19, 0.42, 0.22] : Array(cols).fill(1 / cols);
  const widths = weights.map((w) => Math.round(TEXT_W_DXA * w));
  return new Table({
    width: { size: TEXT_W_DXA, type: WidthType.DXA },
    columnWidths: widths,
    rows: body.map((cells, r) => new TableRow({
      tableHeader: r === 0,
      children: cells.map((c, i) => new TableCell({
        width: { size: widths[i], type: WidthType.DXA },
        shading: r === 0 ? { type: ShadingType.CLEAR, fill: "EDEDED", color: "auto" } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ children: runs(c, { size: 20, bold: r === 0 }), spacing: { line: 276 } })],
      })),
    })),
  });
}

function parse(md, baseDir) {
  const lines = md.split("\n");
  const blocks = [];
  let buf = [];
  const flush = () => { if (buf.length) { blocks.push(paragraph(buf.join(""))); buf = []; } };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^# /.test(line)) { flush(); blocks.push(heading(line.slice(2), 1)); }
    else if (/^## /.test(line)) { flush(); blocks.push(heading(line.slice(3), 2)); }
    else if (/^> /.test(line)) { flush(); blocks.push(note(line.slice(2))); }
    else if (/^!\[/.test(line)) {
      flush();
      const m = line.match(/^!\[([^\]]*)\]\(([^)]+)\)/);
      blocks.push(image(m[2], m[1], baseDir));
    } else if (/^\|/.test(line)) {
      flush();
      const tbl = [];
      while (i < lines.length && /^\|/.test(lines[i])) tbl.push(lines[i++]);
      i--;
      blocks.push(table(tbl));
      blocks.push(new Paragraph({ spacing: { after: 120 } }));
    } else if (line.trim() === "") flush();
    else buf.push(line);
  }
  flush();
  return blocks;
}

const md = fs.readFileSync(inputPath, "utf8");
const doc = new Document({
  styles: {
    default: { document: { run: { font: { ascii: LATIN_FONT, hAnsi: LATIN_FONT, eastAsia: CJK_FONT }, size: BODY_PT * 2 } } },
    paragraphStyles: [
      { id: "Heading1", name: "heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: { ascii: LATIN_FONT, hAnsi: LATIN_FONT, eastAsia: CJK_HEADING_FONT } } },
      { id: "Heading2", name: "heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: { ascii: LATIN_FONT, hAnsi: LATIN_FONT, eastAsia: CJK_HEADING_FONT } } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W_DXA, height: PAGE_H_DXA }, margin: { top: MARGIN_DXA, bottom: MARGIN_DXA, left: MARGIN_DXA, right: MARGIN_DXA } } },
    children: parse(md, path.dirname(inputPath)),
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, buf);
  console.log(`wrote ${outputPath} (${buf.length} bytes)`);
});
