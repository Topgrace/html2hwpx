import os
import tempfile
import re
from pathlib import Path
from typing import List, Tuple, Union

from bs4 import BeautifulSoup, NavigableString, Tag
from latex2mathml.converter import convert as latex_to_mathml
from pyhwpx import Hwp

HTML_PATH = Path("unit1_sample.html")
OUTPUT_FULL = Path("unit1_sample.hwpx").resolve()

MATH_PATTERN = re.compile(r'(\$[^$]+\$|\\\([^\\)]+\\\)|\\\[[^\\]]+\\\])')
Segment = Union[Tuple[str, str], Tuple[str, str, str]]


def split_text_with_math(text: str) -> List[Segment]:
    """텍스트에서 일반 문자열과 LaTeX 수식을 분리한다."""
    parts: List[Segment] = []
    for chunk in MATH_PATTERN.split(text):
        if not chunk:
            continue
        if MATH_PATTERN.fullmatch(chunk):
            latex = chunk
            if latex.startswith("$") and latex.endswith("$"):
                latex = latex[1:-1]
            elif latex.startswith(r"\(") and latex.endswith(r"\)"):
                latex = latex[2:-2]
            elif latex.startswith(r"\[") and latex.endswith(r"\]"):
                latex = latex[2:-2]
            latex = latex.strip()
            if not latex:
                continue
            try:
                mathml = latex_to_mathml(latex)
            except Exception as exc:
                mathml = ""
                print(f"[경고] MathML 변환 실패: {latex} -> {exc}")
            parts.append(("math", latex, mathml))
        else:
            clean = chunk.strip()
            if clean:
                parts.append(("text", clean))
    return parts


def collect_segments(node: Tag) -> List[Segment]:
    """타그 트리를 순회하며 텍스트, 수식, 줄바꿈 세그먼트를 구성한다."""
    segments: List[Segment] = []
    block_level_tags = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
    
    prev_was_block = False
    for child in node.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:  # 공백이 아닌 실제 텍스트만 처리
                segments.extend(split_text_with_math(text))
                prev_was_block = False
            # 공백만 있는 경우 prev_was_block 유지 (블록 간 공백 무시)
        elif isinstance(child, Tag):
            if child.name == "br":
                segments.append(("break", ""))
                prev_was_block = False
            elif child.name in block_level_tags:
                # 블록 레벨 태그: 형제 블록 사이에 줄바꿈 추가
                if prev_was_block:
                    segments.append(("break", ""))
                
                # 블록 내용 추가 (재귀 호출)
                segments.extend(collect_segments(child))
                prev_was_block = True
            else:
                # 인라인 태그: 재귀적으로 처리
                segments.extend(collect_segments(child))
                prev_was_block = False
    
    return [seg for seg in segments if seg]


def extract_blocks(page_limit: Union[int, None] = None):
    """본문에서 제목·문단·인용 블록을 순회하며 (태그, 세그먼트 목록)을 구성한다."""
    soup = BeautifulSoup(HTML_PATH.read_text(encoding="utf-8"), "lxml")
    page_sections = soup.select(".page-content") or [soup.body]

    blocks = []
    for page_idx, main in enumerate(page_sections):
        if page_limit is not None and page_idx >= page_limit:
            break
        
        # page-content의 직접 자식 노드들을 순회
        for child in main.children:
            if isinstance(child, NavigableString):
                continue
            elif isinstance(child, Tag):
                # section과 형제인 br 태그는 빈 줄로 처리
                if child.name == "br":
                    blocks.append(("section-br", []))
                # section 내부의 콘텐츠 블록들을 추출
                elif child.name == "section":
                    for node in child.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"], recursive=True):
                        if node.name == "p" and node.find_parent("blockquote"):
                            continue  # blockquote 내부의 p는 중복이므로 건너뛰는다.
                        segments = collect_segments(node)
                        if segments:
                            blocks.append((node.name, segments))
                # section이 없는 경우를 위해 직접 블록 요소도 처리
                elif child.name in ["h1", "h2", "h3", "h4", "p", "li", "blockquote"]:
                    if child.name == "p" and child.find_parent("blockquote"):
                        continue
                    segments = collect_segments(child)
                    if segments:
                        blocks.append((child.name, segments))
    return blocks


def insert_segments_into_hwp(hwp: Hwp, segments: List[Segment]) -> None:
    """세그먼트를 순회하며 텍스트와 수식을 한글 문서에 삽입한다."""
    for segment in segments:
        kind = segment[0]
        if kind == "text":
            hwp.insert_text(segment[1] + " ")
        elif kind == "break":
            hwp.BreakPara()  # 단순 줄바꿈만 수행
        elif kind == "math":
            _, latex, mathml = segment
            if not mathml:
                hwp.insert_text(f"[수식 변환 실패: {latex}]")
                continue

            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".mml", encoding="utf-8") as temp_mml:
                temp_mml.write(mathml)
                temp_path = temp_mml.name
            try:
                hwp.import_mathml(temp_path)
                # 수식 선택 해제: ESC 키로 선택 해제 후 오른쪽 화살표로 이동
                hwp.HAction.Run("Escape")
                hwp.Run("MoveLineEnd")  # 라인 끝으로 이동
            except Exception as exc:
                print(f"[경고] MathML 삽입 실패: {latex} -> {exc}")
                hwp.insert_text(f"[수식 삽입 실패: {latex}]")
            finally:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass


def create_blockquote_table(hwp: Hwp) -> None:
    """blockquote용 1x1 표를 생성한다."""
    hwp.HAction.GetDefault("TableCreate", hwp.HParameterSet.HTableCreation.HSet)
    hwp.HParameterSet.HTableCreation.Rows = 1
    hwp.HParameterSet.HTableCreation.Cols = 1
    hwp.HParameterSet.HTableCreation.WidthType = 2
    hwp.HParameterSet.HTableCreation.HeightType = 0
    hwp.HParameterSet.HTableCreation.WidthValue = hwp.MiliToHwpUnit(74.0)
    hwp.HParameterSet.HTableCreation.HeightValue = hwp.MiliToHwpUnit(22.6)
    hwp.HParameterSet.HTableCreation.CreateItemArray("ColWidth", 1)
    hwp.HParameterSet.HTableCreation.ColWidth.SetItem(0, hwp.MiliToHwpUnit(70.4))
    hwp.HParameterSet.HTableCreation.CreateItemArray("RowHeight", 1)
    hwp.HParameterSet.HTableCreation.RowHeight.SetItem(0, hwp.MiliToHwpUnit(0.0))
    hwp.HParameterSet.HTableCreation.TableProperties.HorzOffset = hwp.MiliToHwpUnit(0.0)
    hwp.HParameterSet.HTableCreation.TableProperties.VertOffset = hwp.MiliToHwpUnit(0.0)
    hwp.HParameterSet.HTableCreation.TableProperties.HorzRelTo = hwp.HorzRel("Para")
    hwp.HParameterSet.HTableCreation.TableProperties.Width = 20978
    hwp.HAction.Execute("TableCreate", hwp.HParameterSet.HTableCreation.HSet)


def create_full_hwpx(blocks, output_path: Path = OUTPUT_FULL) -> None:
    """HTML에서 추출한 블록을 HWPX 문서로 저장한다."""
    hwp = Hwp(visible=False)
    hwp.set_message_box_mode(0)
    hwp.HAction.Run("FileNew")
    
    # 편집용지 설정 (좌우 여백 25mm)
    hwp.HAction.GetDefault("PageSetup", hwp.HParameterSet.HSecDef.HSet)
    hwp.HParameterSet.HSecDef.PageDef.LeftMargin = hwp.MiliToHwpUnit(25.0)
    hwp.HParameterSet.HSecDef.PageDef.RightMargin = hwp.MiliToHwpUnit(25.0)
    hwp.HParameterSet.HSecDef.HSet.SetItem("ApplyClass", 24)
    hwp.HParameterSet.HSecDef.HSet.SetItem("ApplyTo", 3)
    hwp.HAction.Execute("PageSetup", hwp.HParameterSet.HSecDef.HSet)
    
    # 2단 레이아웃 설정 (구분선 포함)
    hwp.HAction.GetDefault("MultiColumn", hwp.HParameterSet.HColDef.HSet)
    hwp.HParameterSet.HColDef.Count = 2
    hwp.HParameterSet.HColDef.SameSize = 1
    hwp.HParameterSet.HColDef.SameGap = hwp.MiliToHwpUnit(8.0)
    hwp.HParameterSet.HColDef.LineType = hwp.HwpLineType("Solid")
    hwp.HParameterSet.HColDef.LineWidth = hwp.HwpLineWidth("0.12mm")
    hwp.HParameterSet.HColDef.HSet.SetItem("ApplyClass", 832)
    hwp.HParameterSet.HColDef.HSet.SetItem("ApplyTo", 6)
    hwp.HAction.Execute("MultiColumn", hwp.HParameterSet.HColDef.HSet)
    
    hwp.Run("MoveDocBegin")

    for idx, (tag, segments) in enumerate(blocks):
        if tag == "section-br":
            hwp.BreakPara()
            hwp.BreakPara()
            continue
        
        is_heading = tag in {"h1", "h2", "h3", "h4"}
        is_blockquote = tag == "blockquote"

        if is_heading and idx > 0:
            hwp.BreakPara()

        if is_heading:
            hwp.set_font(Bold=True)

        # blockquote는 표 안에 내용 삽입
        if is_blockquote:
            create_blockquote_table(hwp)
            insert_segments_into_hwp(hwp, segments)
            hwp.HAction.Run("MoveTopLevelEnd")  # 표 밖으로 나가기
        else:
            insert_segments_into_hwp(hwp, segments)
            hwp.BreakPara()

        if is_heading:
            hwp.set_font(Bold=False)

    hwp.save_as(str(output_path), format="HWPX")
    hwp.quit()
    print(f"[완료] {output_path} 저장")


if __name__ == "__main__":
    blocks = extract_blocks(page_limit=1) 
    create_full_hwpx(blocks)