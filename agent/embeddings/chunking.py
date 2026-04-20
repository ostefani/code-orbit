import ast
import hashlib
from pathlib import Path

from .types import CodeChunk


DEFAULT_CHUNK_CHAR_LIMIT = 4_000


def _make_chunk(
    path: Path,
    index: int,
    content: str,
    start_line: int,
    end_line: int,
) -> CodeChunk:
    normalized_content = content.strip()
    return CodeChunk(
        path=str(path),
        index=index,
        content=normalized_content,
        start_line=start_line,
        end_line=end_line,
        content_hash=hashlib.sha256(normalized_content.encode("utf-8")).hexdigest(),
    )


def _chunk_by_lines(path: Path, content: str, max_chunk_chars: int) -> list[CodeChunk]:
    lines = content.splitlines()
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    buffer: list[str] = []
    start_line = 1
    current_chars = 0
    index = 0

    for line_number, line in enumerate(lines, start=1):
        line_length = len(line) + 1
        if buffer and current_chars + line_length > max_chunk_chars:
            chunk_text = "\n".join(buffer).strip()
            if chunk_text:
                chunks.append(
                    _make_chunk(
                        path=path,
                        index=index,
                        content=chunk_text,
                        start_line=start_line,
                        end_line=line_number - 1,
                    )
                )
                index += 1
            buffer = []
            current_chars = 0
            start_line = line_number

        buffer.append(line)
        current_chars += line_length

    if buffer:
        chunk_text = "\n".join(buffer).strip()
        if chunk_text:
            chunks.append(
                _make_chunk(
                    path=path,
                    index=index,
                    content=chunk_text,
                    start_line=start_line,
                    end_line=len(lines),
                )
            )

    return chunks


def _chunk_python_source(path: Path, content: str, max_chunk_chars: int) -> list[CodeChunk]:
    lines = content.splitlines()
    try:
        module = ast.parse(content)
    except SyntaxError:
        return _chunk_by_lines(path, content, max_chunk_chars)

    spans: list[tuple[int, int]] = []
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = getattr(node, "end_lineno", node.lineno)
            spans.append((node.lineno, end_line))

    if not spans:
        return _chunk_by_lines(path, content, max_chunk_chars)

    chunks: list[CodeChunk] = []
    current_index = 0
    last_end = 0

    for start_line, end_line in spans:
        if start_line - 1 > last_end:
            gap_text = "\n".join(lines[last_end : start_line - 1]).strip()
            if gap_text:
                chunks.append(
                    _make_chunk(
                        path=path,
                        index=current_index,
                        content=gap_text,
                        start_line=last_end + 1,
                        end_line=start_line - 1,
                    )
                )
                current_index += 1

        block_text = "\n".join(lines[start_line - 1 : end_line]).strip()
        if block_text:
            chunks.append(
                _make_chunk(
                    path=path,
                    index=current_index,
                    content=block_text,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            current_index += 1
        last_end = max(last_end, end_line)

    if last_end < len(lines):
        tail_text = "\n".join(lines[last_end:]).strip()
        if tail_text:
            chunks.append(
                _make_chunk(
                    path=path,
                    index=current_index,
                    content=tail_text,
                    start_line=last_end + 1,
                    end_line=len(lines),
                )
            )

    return chunks or _chunk_by_lines(path, content, max_chunk_chars)


def chunk_file(
    path: Path,
    content: str,
    max_chunk_chars: int = DEFAULT_CHUNK_CHAR_LIMIT,
) -> list[CodeChunk]:
    if not content.strip():
        return []

    if len(content) <= max_chunk_chars:
        return [
            _make_chunk(
                path=path,
                index=0,
                content=content.strip(),
                start_line=1,
                end_line=max(1, content.count("\n") + 1),
            )
        ]

    if path.suffix == ".py":
        return _chunk_python_source(path, content, max_chunk_chars)

    return _chunk_by_lines(path, content, max_chunk_chars)
