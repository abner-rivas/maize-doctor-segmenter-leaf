#!/usr/bin/env python3
# ruff: noqa: E501, I001
"""Build printable HTML and LaTeX from the report's constrained Markdown."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SECTIONS = ROOT / "sections"
AUTHORS = [
    ("Josias Abner Rivas Fuentes", "RF20010"),
    ("David Alejandro Deras Cerros", "DC19019"),
    ("Elmer Edenilson Rosales Molina", "RM20001"),
]
REFERENCES = {
    "jocher2026yolo26": {
        "number": 1,
        "html": "G. Jocher et al., <em>Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models</em>, arXiv:2606.03748, 2026.",
        "url": "https://arxiv.org/abs/2606.03748",
    },
    "ultralytics2026segment": {
        "number": 2,
        "html": "Ultralytics, <em>Instance Segmentation with Ultralytics YOLO</em>, documentación oficial, consultada el 10 de septiembre de 2026.",
        "url": "https://docs.ultralytics.com/tasks/segment/",
    },
    "lin2014coco": {
        "number": 3,
        "html": "T.-Y. Lin et al., <em>Microsoft COCO: Common Objects in Context</em>, ECCV, 2014.",
        "url": "https://arxiv.org/abs/1405.0312",
    },
    "dice1945": {
        "number": 4,
        "html": "L. R. Dice, <em>Measures of the Amount of Ecologic Association Between Species</em>, Ecology 26(3), 1945.",
        "url": "https://doi.org/10.2307/1932409",
    },
    "mitchell2019modelcards": {
        "number": 5,
        "html": "M. Mitchell et al., <em>Model Cards for Model Reporting</em>, FAT*, 2019.",
        "url": "https://doi.org/10.1145/3287560.3287596",
    },
    "gebru2021datasheets": {
        "number": 6,
        "html": "T. Gebru et al., <em>Datasheets for Datasets</em>, Communications of the ACM 64(12), 2021.",
        "url": "https://doi.org/10.1145/3458723",
    },
    "huggingface2026dataset": {
        "number": 7,
        "html": "daiv05, <em>corn-leaf-diseases-pests-and-deficiencies</em>, Hugging Face Datasets, commit e515ab2f, consultado el 10 de septiembre de 2026.",
        "url": "https://huggingface.co/datasets/daiv05/corn-leaf-diseases-pests-and-deficiencies/tree/e515ab2f1e4c5729f8447520f1a630cf14c532dc",
    },
}


@dataclass
class Block:
    kind: str
    value: object
    extra: object | None = None


@dataclass
class Document:
    title: str
    slug: str
    blocks: list[Block]
    numbered: bool
    number: int | None = None


def slugify(value: str) -> str:
    replacements = str.maketrans("áéíóúñÁÉÍÓÚÑ", "aeiounAEIOUN")
    value = value.translate(replacements).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "seccion"


def parse_markdown(path: Path) -> Document:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path} debe comenzar con un titulo de nivel 1")
    title = lines[0][2:].strip()
    blocks: list[Block] = []
    index = 1
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(Block("paragraph", " ".join(item.strip() for item in paragraph)))
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped == "<!-- pagebreak -->":
            flush_paragraph()
            blocks.append(Block("pagebreak", ""))
            index += 1
            continue
        if stripped.startswith("## ") or stripped.startswith("### "):
            flush_paragraph()
            level = 3 if stripped.startswith("### ") else 2
            blocks.append(Block("heading", stripped[level + 1 :], level))
            index += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped[3:].strip()
            index += 1
            code: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError(f"Bloque de codigo sin cerrar en {path}")
            blocks.append(Block("code", "\n".join(code), language))
            index += 1
            continue
        if stripped.startswith("::: "):
            flush_paragraph()
            match = re.match(r"::: (evidence|risk|note)(?:\s+(.*))?$", stripped)
            if not match:
                raise ValueError(f"Callout invalido en {path}: {stripped}")
            kind, title_value = match.group(1), match.group(2) or ""
            index += 1
            body: list[str] = []
            while index < len(lines) and lines[index].strip() != ":::":
                body.append(lines[index].strip())
                index += 1
            if index == len(lines):
                raise ValueError(f"Callout sin cerrar en {path}")
            blocks.append(Block("callout", " ".join(body), (kind, title_value)))
            index += 1
            continue
        image_match = re.fullmatch(r"!\[(.+)]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            blocks.append(Block("figure", image_match.group(2), image_match.group(1)))
            index += 1
            continue
        if stripped.startswith("Table: "):
            flush_paragraph()
            caption = stripped[7:].strip()
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            if len(table_lines) < 2:
                raise ValueError(f"Tabla sin contenido en {path}: {caption}")
            rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            if all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
                rows.pop(1)
            blocks.append(Block("table", rows, caption))
            continue
        if re.match(r"^[-*] ", stripped) or re.match(r"^\d+\. ", stripped):
            flush_paragraph()
            ordered = bool(re.match(r"^\d+\. ", stripped))
            items: list[str] = []
            pattern = r"^\d+\. " if ordered else r"^[-*] "
            while index < len(lines) and re.match(pattern, lines[index].strip()):
                items.append(re.sub(pattern, "", lines[index].strip(), count=1))
                index += 1
            blocks.append(Block("list", items, ordered))
            continue
        paragraph.append(line)
        index += 1
    flush_paragraph()
    numbered = title.lower() not in {"resumen", "referencias"}
    return Document(title=title, slug=slugify(title), blocks=blocks, numbered=numbered)


def inline_html(value: str) -> str:
    value = html.escape(value, quote=False)
    code_tokens: dict[str, str] = {}

    def code_repl(match: re.Match[str]) -> str:
        token = f"@@CODE{len(code_tokens)}@@"
        code_tokens[token] = f"<code>{match.group(1)}</code>"
        return token

    value = re.sub(r"`([^`]+)`", code_repl, value)

    def citation_repl(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(";")]
        links = []
        for key in keys:
            reference = REFERENCES.get(key)
            if reference is None:
                raise KeyError(f"Referencia desconocida: {key}")
            links.append(f'<a class="citation" href="#ref-{key}">{reference["number"]}</a>')
        return "[" + ", ".join(links) + "]"

    value = re.sub(r"\[@([^]]+)]", citation_repl, value)
    value = re.sub(r"\[([^]]+)]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", value)
    for token, replacement in code_tokens.items():
        value = value.replace(token, replacement)
    return value


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def inline_tex(value: str) -> str:
    tokens: dict[str, str] = {}

    def token(replacement: str) -> str:
        key = f"ZZTOKEN{len(tokens)}ZZ"
        tokens[key] = replacement
        return key

    value = re.sub(
        r"`([^`]+)`",
        lambda match: token(r"\texttt{" + tex_escape(match.group(1)) + "}"),
        value,
    )

    def citation_repl(match: re.Match[str]) -> str:
        keys = ",".join(key.strip() for key in match.group(1).split(";"))
        return token(r"\citep{" + keys + "}")

    value = re.sub(r"\[@([^]]+)]", citation_repl, value)

    def link_repl(match: re.Match[str]) -> str:
        return token(r"\href{" + match.group(2) + "}{" + tex_escape(match.group(1)) + "}")

    value = re.sub(r"\[([^]]+)]\(([^)]+)\)", link_repl, value)
    value = tex_escape(value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", value)
    for key, replacement in tokens.items():
        value = value.replace(key, replacement)
    return value


def html_documents(documents: list[Document]) -> str:
    figure_count = 0
    table_count = 0
    figure_list: list[tuple[int, str]] = []
    table_list: list[tuple[int, str]] = []
    section_number = 0
    for document in documents:
        if document.numbered:
            section_number += 1
            document.number = section_number
        for block in document.blocks:
            if block.kind == "figure":
                figure_count += 1
                block.extra = (figure_count, str(block.extra))
                figure_list.append((figure_count, str(block.extra[1])))
            elif block.kind == "table":
                table_count += 1
                block.extra = (table_count, str(block.extra))
                table_list.append((table_count, str(block.extra[1])))

    cover = f"""
<section class="cover">
  <div class="university">UNIVERSIDAD DE EL SALVADOR</div>
  <div class="faculty">FACULTAD DE INGENIERÍA Y ARQUITECTURA</div>
  <div class="faculty">ESCUELA DE INGENIERÍA DE SISTEMAS INFORMÁTICOS</div>
  <img src="figures/logo.png" alt="Logo Doctor Maíz">
  <h1>DOCTOR MAÍZ</h1>
  <div class="subtitle">Segmentación de instancias de hojas de maíz para aislamiento de tejido foliar</div>
  <div class="report-kind">INFORME TÉCNICO DEL SEGMENTADOR</div>
  <div class="subtitle">Datos, entrenamiento, evaluación e integración reproducible</div>
  <table class="authors">{''.join(f'<tr><td><strong>{name}</strong></td><td>{identifier}</td></tr>' for name, identifier in AUTHORS)}</table>
  <div class="place">San Salvador, El Salvador<br>Septiembre de 2026</div>
</section>
"""
    toc = "".join(
        f'<li><a href="#{document.slug}">{html.escape(document.title)}</a></li>'
        for document in documents
        if document.numbered
    )
    figures_html = "".join(f"<li>Figura {number}. {inline_html(caption)}</li>" for number, caption in figure_list)
    tables_html = "".join(f"<li>Tabla {number}. {inline_html(caption)}</li>" for number, caption in table_list)
    front = f"""
<section class="front-matter toc"><h1>Índice</h1><ol>{toc}</ol></section>
<section class="front-matter figure-list"><h1>Lista de figuras</h1><ol>{figures_html}</ol></section>
<section class="front-matter table-list"><h1>Lista de tablas</h1><ol>{tables_html}</ol></section>
"""

    rendered: list[str] = []
    for document in documents:
        title = f"{document.number}. {document.title}" if document.numbered else document.title
        rendered.append(f'<section id="{document.slug}" class="report-section">')
        rendered.append(f'<h1 class="section-title">{html.escape(title)}</h1>')
        for block in document.blocks:
            if block.kind == "paragraph":
                rendered.append(f"<p>{inline_html(str(block.value))}</p>")
            elif block.kind == "heading":
                level = int(block.extra or 2)
                rendered.append(f"<h{level}>{inline_html(str(block.value))}</h{level}>")
            elif block.kind == "list":
                tag = "ol" if block.extra else "ul"
                rendered.append(f"<{tag}>" + "".join(f"<li>{inline_html(item)}</li>" for item in block.value) + f"</{tag}>")
            elif block.kind == "code":
                rendered.append(f'<pre><code>{html.escape(str(block.value))}</code></pre>')
            elif block.kind == "figure":
                number, caption = block.extra
                rendered.append(
                    f'<figure class="figure" id="fig-{number}"><img src="{html.escape(str(block.value))}" alt="{html.escape(caption)}">'
                    f'<figcaption class="caption"><span class="label">Figura {number}.</span> {inline_html(caption)}</figcaption></figure>'
                )
            elif block.kind == "table":
                number, caption = block.extra
                rows = block.value
                head = "".join(f"<th>{inline_html(cell)}</th>" for cell in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{inline_html(cell)}</td>" for cell in row) + "</tr>" for row in rows[1:])
                rendered.append(
                    f'<div class="table-wrap" id="tab-{number}"><div class="table-caption"><span class="label">Tabla {number}.</span> {inline_html(caption)}</div>'
                    f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
                )
            elif block.kind == "callout":
                kind, callout_title = block.extra
                css = "evidence" if kind == "evidence" else "risk" if kind == "risk" else "note"
                prefix = f"<strong>{inline_html(callout_title)}</strong> " if callout_title else ""
                rendered.append(f'<div class="callout {css}">{prefix}{inline_html(str(block.value))}</div>')
            elif block.kind == "pagebreak":
                rendered.append('<div class="page-break"></div>')
        if document.title.lower() == "resumen":
            rendered.append('<p class="keywords"><strong>Palabras clave:</strong> segmentación de instancias, hojas de maíz, YOLO26, quality gate, trazabilidad, inferencia reproducible.</p>')
        rendered.append("</section>")
        if document.title.lower() == "resumen":
            rendered.append(front)

    references_html = "".join(
        f'<li id="ref-{key}">{reference["html"]} <a href="{reference["url"]}">{reference["url"]}</a></li>'
        for key, reference in sorted(REFERENCES.items(), key=lambda item: item[1]["number"])
    )
    rendered.append(f'<section id="referencias" class="report-section references"><h1 class="section-title">Referencias</h1><ol>{references_html}</ol></section>')
    return cover + "\n".join(rendered)


def tex_documents(documents: list[Document]) -> str:
    rendered: list[str] = []
    for document in documents:
        command = r"\section*" if not document.numbered else r"\section"
        rendered.append(f"{command}{{{inline_tex(document.title)}}}")
        if not document.numbered:
            rendered.append(r"\addcontentsline{toc}{section}{" + inline_tex(document.title) + "}")
        for block in document.blocks:
            if block.kind == "paragraph":
                rendered.append(inline_tex(str(block.value)) + "\n")
            elif block.kind == "heading":
                command = r"\subsubsection" if int(block.extra or 2) == 3 else r"\subsection"
                rendered.append(f"{command}{{{inline_tex(str(block.value))}}}")
            elif block.kind == "list":
                environment = "enumerate" if block.extra else "itemize"
                rendered.append(r"\begin{" + environment + "}")
                rendered.extend(r"\item " + inline_tex(item) for item in block.value)
                rendered.append(r"\end{" + environment + "}")
            elif block.kind == "code":
                rendered.append(r"\begin{verbatim}" + "\n" + str(block.value) + "\n" + r"\end{verbatim}")
            elif block.kind == "figure":
                _number, caption = block.extra
                rendered.append(
                    "\n".join(
                        [
                            r"\begin{figure}[H]",
                            r"\centering",
                            r"\includegraphics[width=0.94\textwidth]{" + str(block.value) + "}",
                            r"\caption{" + inline_tex(caption) + "}",
                            r"\end{figure}",
                        ]
                    )
                )
            elif block.kind == "table":
                _number, caption = block.extra
                rows = block.value
                columns = len(rows[0])
                spec = "l" + "r" * (columns - 1)
                lines = [r"\begin{table}[H]", r"\centering", r"\caption{" + inline_tex(caption) + "}", r"\small"]
                if columns > 5:
                    lines.append(r"\resizebox{\textwidth}{!}{%")
                lines.extend([r"\begin{tabular}{" + spec + "}", r"\toprule"])
                lines.append(" & ".join(inline_tex(cell) for cell in rows[0]) + r" \\")
                lines.append(r"\midrule")
                for row in rows[1:]:
                    lines.append(" & ".join(inline_tex(cell) for cell in row) + r" \\")
                lines.extend([r"\bottomrule", r"\end{tabular}"])
                if columns > 5:
                    lines.append("}")
                lines.append(r"\end{table}")
                rendered.append("\n".join(lines))
            elif block.kind == "callout":
                kind, callout_title = block.extra
                label = inline_tex(callout_title) if callout_title else kind.capitalize()
                rendered.append(r"\begin{quote}\textbf{" + label + ":} " + inline_tex(str(block.value)) + r"\end{quote}")
            elif block.kind == "pagebreak":
                rendered.append(r"\clearpage")
        if document.title.lower() == "resumen":
            rendered.append(r"\noindent\textbf{Palabras clave:} segmentación de instancias, hojas de maíz, YOLO26, quality gate, trazabilidad, inferencia reproducible.")
    return "\n\n".join(rendered)


def build() -> None:
    paths = sorted(SECTIONS.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No hay secciones en {SECTIONS}")
    documents = [parse_markdown(path) for path in paths]
    body = html_documents(documents)
    page = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Doctor Maíz - Informe técnico del segmentador</title><link rel="stylesheet" href="report.css"></head><body>{body}</body></html>
"""
    (ROOT / "main.html").write_text(page, encoding="utf-8")

    tex_body = tex_documents(documents)
    author_rows = (" " + r"\\" + "\n").join(
        r"\textbf{" + tex_escape(name) + "} & " + identifier
        for name, identifier in AUTHORS
    )
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-nodecimaldot]{babel}
\usepackage{lmodern,geometry,graphicx,booktabs,array,tabularx,longtable,float}
\usepackage{microtype,setspace,enumitem,fancyhdr,xcolor}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\geometry{top=2.1cm,bottom=2.2cm,left=2.25cm,right=2.25cm}
\setstretch{1.20}\setlength{\parindent}{1.1em}\setlength{\parskip}{0.25em}
\setlength{\headheight}{14pt}\setlength{\emergencystretch}{2em}
\graphicspath{{figures/}}\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{Doctor Maíz}\fancyhead[R]{Informe del segmentador}\fancyfoot[C]{\thepage}
\hypersetup{pdftitle={Doctor Maíz — Informe técnico del segmentador}}
\begin{document}
\begin{titlepage}\thispagestyle{empty}\begin{center}
{\Large\bfseries UNIVERSIDAD DE EL SALVADOR}\\[0.2cm]
{\large FACULTAD DE INGENIERÍA Y ARQUITECTURA}\\[0.15cm]
{\large ESCUELA DE INGENIERÍA DE SISTEMAS INFORMÁTICOS}\\[0.7cm]
\includegraphics[width=0.20\textwidth]{logo.png}\\[0.7cm]
{\LARGE\bfseries DOCTOR MAÍZ}\\[0.3cm]
{\Large Segmentación de instancias de hojas de maíz\\para aislamiento de tejido foliar}\\[0.7cm]
{\LARGE\bfseries INFORME TÉCNICO DEL SEGMENTADOR}\\[0.2cm]
{\large Datos, entrenamiento, evaluación e integración reproducible}\\[1cm]
\begin{tabular}{ll}
""" + author_rows + r"""
\end{tabular}\vfill
{\large San Salvador, El Salvador}\\[0.15cm]{\large Septiembre de 2026}
\end{center}\end{titlepage}
\pagenumbering{roman}
""" + tex_body.split(r"\section{", 1)[0] + r"""
\clearpage\tableofcontents\clearpage\listoffigures\clearpage\listoftables\clearpage
\pagenumbering{arabic}
\section{""" + tex_body.split(r"\section{", 1)[1] + r"""
\clearpage\bibliographystyle{plainnat}\bibliography{referencias}
\end{document}
"""
    (ROOT / "main.tex").write_text(tex, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source_sections": [path.relative_to(ROOT).as_posix() for path in paths],
        "figures": sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "figures").glob("*.png")),
        "outputs": ["main.html", "main.tex", "main.pdf"],
    }
    (ROOT / "MANIFEST.generated.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Built main.html and main.tex from {len(paths)} source sections")


def print_pdf(chrome: str) -> None:
    output = ROOT / "main.pdf"
    command = [
        chrome,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output}",
        (ROOT / "main.html").as_uri(),
    ]
    subprocess.run(command, check=True)
    if not output.is_file() or output.stat().st_size < 10_000:
        raise RuntimeError("Chrome no produjo un PDF valido")
    print(f"Printed {output} ({output.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true", help="Print main.pdf after building")
    parser.add_argument("--chrome", default="google-chrome")
    args = parser.parse_args()
    build()
    if args.pdf:
        print_pdf(args.chrome)


if __name__ == "__main__":
    main()
