"""
Tests para formato de salida de generar_tabla_favoritos2.py.

Valida que:
1. Section headers tienen newlines propios (no concatenados).
2. Rivales Comunes estan ordenados por fecha (mas antigua primero).
3. No hay literales \\1 o \\n doble-escaped en el output.
4. El source code no tiene f.write("---...") sin \\n al final.
"""
import os
import re
import pytest
import ast

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analisis_partidos_pandas.txt",
)

SOURCE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "generar_tabla_favoritos2.py",
)

# Headers that must appear on their own line
EXPECTED_HEADERS_LITERAL = [
    "--- METADATOS DEL ANALISIS ---",
    "--- RESUMEN DE PREDICCION ---",
    "--- PREDICCION DE SETS Y GAMES ---",
    "--- DISTRIBUCION DE PESOS DEL ANALISIS ---",
    "--- RAZONAMIENTO CLAVE Y LOGS DE PREDICCION ---",
    "--- Logica de Ponderacion y Scores Detallados ---",
    "--- Enfrentamientos Directos ---",
    "--- Analisis de Forma Reciente ---",
    "--- ANALISIS POR SUPERFICIE ---",
    "--- ANALISIS POR UBICACION ---",
    "--- Patrones Clave en Historial ---",
]

# Regex patterns for headers with variable content
EXPECTED_HEADERS_REGEX = [
    r"--- Rivales Comunes \(Total: \d+\) ---",
    r"--- Justificaci.n Detallada de Ventaja \(Rivales Comunes\) ---",
    r"--- Historial Detallado de .+ ---",
]


def _read_output():
    """Read the output file, skip if missing or empty."""
    if not os.path.exists(OUTPUT_FILE):
        pytest.skip(f"Output file not found: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        pytest.skip("Output file is empty")
    return content


def _read_source():
    """Read the source file."""
    if not os.path.exists(SOURCE_FILE):
        pytest.skip(f"Source file not found: {SOURCE_FILE}")
    with open(SOURCE_FILE, encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Section headers have proper newlines
# ─────────────────────────────────────────────────────────────────────────────


class TestSectionHeaderNewlines:
    """Every --- HEADER --- must be on its own line: preceded and followed by newline."""

    def setup_method(self):
        self.content = _read_output()

    def _assert_header_isolated(self, pattern, is_regex=False):
        """Find all occurrences of a header and verify each is on its own line."""
        if is_regex:
            matches = list(re.finditer(pattern, self.content))
        else:
            # Normalize accents for matching against actual content
            matches = list(re.finditer(re.escape(pattern), self.content))

        if not matches:
            return  # Header not present in this file -- not an error

        for m in matches:
            start, end = m.start(), m.end()

            # Character before must be \n or start-of-file
            if start > 0:
                before = self.content[start - 1]
                assert before == "\n", (
                    f"Header '{m.group()}' at offset {start} is NOT preceded by newline. "
                    f"Found {repr(before)} instead. "
                    f"Context: ...{repr(self.content[max(0, start-30):end+30])}..."
                )

            # Character after must be \n or end-of-file
            if end < len(self.content):
                after = self.content[end]
                assert after == "\n", (
                    f"Header '{m.group()}' at offset {start} is NOT followed by newline. "
                    f"Found {repr(after)} instead. "
                    f"Context: ...{repr(self.content[max(0, start-30):end+30])}..."
                )

    def test_header_metadatos(self):
        self._assert_header_isolated("--- METADATOS DEL AN\u00c1LISIS ---")

    def test_header_resumen_prediccion(self):
        self._assert_header_isolated("--- RESUMEN DE PREDICCI\u00d3N ---")

    def test_header_prediccion_sets_games(self):
        self._assert_header_isolated("--- PREDICCI\u00d3N DE SETS Y GAMES ---")

    def test_header_distribucion_pesos(self):
        self._assert_header_isolated("--- DISTRIBUCI\u00d3N DE PESOS DEL AN\u00c1LISIS ---")

    def test_header_razonamiento(self):
        self._assert_header_isolated("--- RAZONAMIENTO CLAVE Y LOGS DE PREDICCI\u00d3N ---")

    def test_header_logica_ponderacion(self):
        self._assert_header_isolated("--- L\u00f3gica de Ponderaci\u00f3n y Scores Detallados ---")

    def test_header_enfrentamientos_directos(self):
        self._assert_header_isolated("--- Enfrentamientos Directos ---")

    def test_header_rivales_comunes(self):
        self._assert_header_isolated(
            r"--- Rivales Comunes \(Total: \d+\) ---", is_regex=True
        )

    def test_header_justificacion_ventaja(self):
        self._assert_header_isolated(
            r"--- Justificaci\u00f3n Detallada de Ventaja \(Rivales Comunes\) ---",
            is_regex=True,
        )

    def test_header_forma_reciente(self):
        self._assert_header_isolated("--- An\u00e1lisis de Forma Reciente ---")

    def test_header_superficie(self):
        self._assert_header_isolated("--- AN\u00c1LISIS POR SUPERFICIE ---")

    def test_header_ubicacion(self):
        self._assert_header_isolated("--- AN\u00c1LISIS POR UBICACI\u00d3N ---")

    def test_header_historial_detallado(self):
        self._assert_header_isolated(r"--- Historial Detallado de .+ ---", is_regex=True)

    def test_header_patrones_historial(self):
        self._assert_header_isolated("--- Patrones Clave en Historial ---")

    def test_header_overs(self):
        self._assert_header_isolated("--- AN\u00c1LISIS DE PROBABILIDAD (OVERS) ---")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Common opponents sorted by date (oldest first)
# ─────────────────────────────────────────────────────────────────────────────


class TestCommonOpponentsSorting:
    """Rows in Rivales Comunes sections must be sorted oldest-first."""

    def setup_method(self):
        self.content = _read_output()

    @staticmethod
    def _parse_date(date_str):
        """Parse DD.MM.YYYY to a comparable tuple (YYYY, MM, DD)."""
        try:
            parts = date_str.strip().split(".")
            if len(parts) != 3:
                return None
            d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
            return (y, m, d)
        except (ValueError, IndexError):
            return None

    def test_common_opponents_sorted_by_oldest_date(self):
        """For each Rivales Comunes section, extract dates and verify oldest-first order."""
        # Split content by Rivales Comunes headers
        sections = re.split(r"--- Rivales Comunes \(Total: \d+\) ---", self.content)
        if len(sections) <= 1:
            pytest.skip("No Rivales Comunes sections found")

        violations = []
        for i, section in enumerate(sections[1:], 1):
            # Take only content up to the next --- header
            next_header = re.search(r"\n---[^|]", section)
            if next_header:
                section = section[: next_header.start()]

            # Extract all DD.MM.YYYY dates from table rows (lines starting with |)
            rows_dates = []
            for line in section.split("\n"):
                if not line.startswith("|") or line.startswith("|:"):
                    continue
                # Find all dates in this row
                dates_in_row = re.findall(r"\d{2}\.\d{2}\.\d{4}", line)
                if not dates_in_row:
                    continue
                parsed = [self._parse_date(d) for d in dates_in_row]
                parsed = [d for d in parsed if d is not None]
                if parsed:
                    rows_dates.append(min(parsed))

            # Check sorting
            if len(rows_dates) >= 2:
                for j in range(len(rows_dates) - 1):
                    if rows_dates[j] > rows_dates[j + 1]:
                        violations.append(
                            f"Section {i}: row {j} date {rows_dates[j]} > row {j+1} date {rows_dates[j+1]}"
                        )

        assert not violations, (
            f"Common opponents not sorted oldest-first:\n" + "\n".join(violations)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. No literal \1 or double-escaped \\n in output
# ─────────────────────────────────────────────────────────────────────────────


class TestNoEscapeArtifacts:
    """Output must not contain literal backslash-1 or double-escaped newlines."""

    def setup_method(self):
        self.content = _read_output()

    def test_no_literal_backslash_1(self):
        """No literal \\1 (regex backreference artifact) in output."""
        occurrences = [
            m.start()
            for m in re.finditer(r"\\1", self.content)
        ]
        if occurrences:
            # Show context around first occurrence
            pos = occurrences[0]
            ctx = self.content[max(0, pos - 30) : pos + 30]
            assert False, (
                f"Found {len(occurrences)} literal '\\1' in output. "
                f"First at offset {pos}: ...{repr(ctx)}..."
            )

    def test_no_double_escaped_newlines(self):
        """No literal \\n (should be actual newlines, not escaped)."""
        # Match literal backslash followed by n, but not inside LOG lines
        # which legitimately show dict repr with \\n
        lines_with_escaped_n = []
        for line_num, line in enumerate(self.content.split("\n"), 1):
            # Skip lines that are log output showing Python dicts/reprs
            if "LOG_" in line or "{'surface" in line or "weights=" in line:
                continue
            if "\\n" in line:
                lines_with_escaped_n.append((line_num, line[:100]))

        assert not lines_with_escaped_n, (
            f"Found {len(lines_with_escaped_n)} lines with literal '\\n'. "
            f"First: line {lines_with_escaped_n[0][0]}: {lines_with_escaped_n[0][1]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Source code validation: f.write("---") must end with \n
# ─────────────────────────────────────────────────────────────────────────────


class TestSourceCodeHeaderWrites:
    """Every f.write containing '---' header must end the string with \\n."""

    def setup_method(self):
        self.source = _read_source()

    def test_all_fwrite_headers_end_with_newline(self):
        """Find f.write calls containing --- HEADER --- and verify they end with \\n."""
        # Match f.write("..." ) or f.write(f"..." ) containing ---
        # We look for the pattern: f.write( followed by a string containing ---
        violations = []
        for line_num, line in enumerate(self.source.split("\n"), 1):
            stripped = line.strip()
            # Must be an f.write call containing a section header pattern
            if "f.write(" not in stripped:
                continue
            if "---" not in stripped:
                continue
            # Skip lines that are comments
            if stripped.startswith("#"):
                continue
            # Skip lines writing table separators (markdown |:---|)
            if "|:" in stripped or "|-" in stripped:
                continue

            # Extract the string argument — check if it ends with \n before closing quote
            # Pattern: the closing --- should be followed by \n" or \n')
            # Bad: f.write("--- HEADER ---") or f.write("--- HEADER ---" + something)
            # Good: f.write("--- HEADER ---\n") or f.write(f"\n--- HEADER ---\n")
            match = re.search(r'f\.write\([f]?["\'](.+?)["\']', stripped)
            if not match:
                continue
            string_content = match.group(1)
            # Must contain a header-like pattern: --- WORD ---
            if not re.search(r'---\s+\S.*?---', string_content):
                continue
            # The string must end with \n (escaped in source as \\n)
            if not string_content.endswith("\\n"):
                violations.append(
                    f"Line {line_num}: {stripped[:120]}"
                )

        assert not violations, (
            f"f.write() calls with --- headers missing trailing \\n:\n"
            + "\n".join(violations)
        )

    def test_no_double_escaped_backslash_n_in_fwrite(self):
        """No f.write containing literal \\\\n (double-escaped, writes literal \\n)."""
        violations = []
        for line_num, line in enumerate(self.source.split("\n"), 1):
            stripped = line.strip()
            if "f.write(" not in stripped:
                continue
            if stripped.startswith("#"):
                continue
            # Check for \\n inside f.write (double backslash + n = literal \n in output)
            # In Python source: '\\n' inside a regular string = literal backslash + n
            # This appears in source as: f.write("\\n--- ...")
            # In the raw file bytes this would be: \\ n
            if '\\\\n' in line or '\\1' in line:
                violations.append(f"Line {line_num}: {stripped[:120]}")

        assert not violations, (
            f"f.write() calls with double-escaped \\\\n or \\1 (writes literal text, not newline):\n"
            + "\n".join(violations)
        )

    def test_no_backslash_1_in_fwrite(self):
        """No f.write containing \\1 (regex backreference artifact)."""
        violations = []
        for line_num, line in enumerate(self.source.split("\n"), 1):
            stripped = line.strip()
            if "f.write(" not in stripped:
                continue
            if stripped.startswith("#"):
                continue
            # \1 in an f.write is always a bug — should be actual header text
            if "\\1" in stripped:
                violations.append(f"Line {line_num}: {stripped[:120]}")

        assert not violations, (
            f"f.write() calls containing \\1 (regex backreference artifact):\n"
            + "\n".join(violations)
        )
