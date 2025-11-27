import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Union

from bs4 import BeautifulSoup, NavigableString, Tag
from pyhwpx import Hwp

MATH_PATTERN = re.compile(r'(\$[^$]+\$|\\\([^\\)]+\\\)|\\\[[^\\]]+\\\])')
Segment = Union[Tuple[str, str], Tuple[str, str, str]]


def find_matching_brace(text: str, start_pos: int) -> int:
    """중괄호 '{' 위치가 주어졌을 때 매칭되는 '}' 위치를 찾는다."""
    count = 1
    pos = start_pos + 1
    while pos < len(text) and count > 0:
        if text[pos] == '{':
            count += 1
        elif text[pos] == '}':
            count -= 1
        pos += 1
    return pos - 1 if count == 0 else -1


def convert_frac_to_hwp(latex: str) -> str:
    """중첩된 \frac를 한글 수식 문법으로 재귀적으로 변환한다."""
    result = []
    i = 0
    while i < len(latex):
        if latex[i:i+5] == r'\frac':
            # \frac 발견
            i += 5
            # 공백 건너뛰기
            while i < len(latex) and latex[i] == ' ':
                i += 1
            
            if i >= len(latex) or latex[i] != '{':
                result.append(r'\frac')
                continue
            
            # 첫 번째 인자 추출 (분자)
            start_pos = i
            end_pos = find_matching_brace(latex, start_pos)
            if end_pos == -1:
                result.append(r'\frac{')
                i += 1
                continue
            
            numerator = latex[start_pos+1:end_pos]
            i = end_pos + 1
            
            # 공백 건너뛰기
            while i < len(latex) and latex[i] == ' ':
                i += 1
            
            if i >= len(latex) or latex[i] != '{':
                result.append(r'\frac{' + numerator + '}')
                continue
            
            # 두 번째 인자 추출 (분모)
            start_pos = i
            end_pos = find_matching_brace(latex, start_pos)
            if end_pos == -1:
                result.append(r'\frac{' + numerator + '}{')
                i += 1
                continue
            
            denominator = latex[start_pos+1:end_pos]
            i = end_pos + 1
            
            # 재귀적으로 변환
            numerator_converted = convert_frac_to_hwp(numerator)
            denominator_converted = convert_frac_to_hwp(denominator)
            
            # 한글 수식 문법으로 변환
            result.append('{' + numerator_converted + '} over {' + denominator_converted + '}')
        else:
            result.append(latex[i])
            i += 1
    
    return ''.join(result)


def latex_to_hwp_equation(latex: str, max_length: int = 50) -> str:
    """
    LaTeX 수식을 한글 수식 문법으로 변환한다.
    
    한글 수식 문법 참고:
    - 분수: a over b
    - 제곱: x^2
    - 아래첨자: x_1
    - 제곱근: sqrt x
    - 적분: int _a ^b f(x)dx
    - 시그마: sum_{i=1} ^{n}
    - 극한: lim _{x -> 0}
    - 괄호: {  } 로 묶음
    - 줄바꿈: # (한글 수식 내 줄바꿈)
    """
    hwp_eq = latex.strip()
    
    # 기본적인 LaTeX 명령어를 한글 수식 문법으로 변환
    
    # 분수: \frac{a}{b} -> {a} over {b} (중첩 지원)
    hwp_eq = convert_frac_to_hwp(hwp_eq)
    
    # 제곱근: \sqrt{x} -> sqrt {x}
    hwp_eq = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt {\1}', hwp_eq)
    hwp_eq = re.sub(r'\\sqrt\[([^]]+)\]\{([^}]+)\}', r'root {\1} of {\2}', hwp_eq)
    
    # 거듭제곱: a^b -> a^{b} (괄호가 없는 단일 문자/숫자 거듭제곱)
    # 이미 중괄호로 감싸진 경우는 제외하고, 단일 문자나 숫자만 중괄호로 감싸기
    hwp_eq = re.sub(r'\^([^{\s])', r'^{\1}', hwp_eq)
    
    # 점 장식: \dot{x} -> dot x
    hwp_eq = re.sub(r'\\dot\{([^}]+)\}', r'dot \1', hwp_eq)
    hwp_eq = re.sub(r'\\ddot\{([^}]+)\}', r'ddot \1', hwp_eq)
    hwp_eq = re.sub(r'\\hat\{([^}]+)\}', r'hat \1', hwp_eq)
    hwp_eq = re.sub(r'\\bar\{([^}]+)\}', r'bar \1', hwp_eq)
    hwp_eq = re.sub(r'\\vec\{([^}]+)\}', r'vec \1', hwp_eq)
    hwp_eq = re.sub(r'\\tilde\{([^}]+)\}', r'tilde \1', hwp_eq)
    
    # 적분: \int -> int
    hwp_eq = hwp_eq.replace(r'\int', 'int')
    
    # 시그마: \sum -> sum
    hwp_eq = hwp_eq.replace(r'\sum', 'sum')
    
    # 곱셈: \times -> times
    hwp_eq = hwp_eq.replace(r'\times', 'times')
    
    # 극한: \lim -> lim
    hwp_eq = hwp_eq.replace(r'\lim', 'lim')
    
    # 화살표
    hwp_eq = hwp_eq.replace(r'\to', '->')
    hwp_eq = hwp_eq.replace(r'\rightarrow', 'rarrow')
    hwp_eq = hwp_eq.replace(r'\Rightarrow', 'RARROW')
    hwp_eq = hwp_eq.replace(r'\implies', 'RARROW')
    hwp_eq = hwp_eq.replace(r'\leftarrow', 'larrow')
    hwp_eq = hwp_eq.replace(r'\Leftarrow', 'LARROW')
    
    # 무한대: \infty -> inf
    hwp_eq = hwp_eq.replace(r'\infty', 'inf')
    
    # 그리스 문자들 (소문자)
    greek_lower = {
        r'\alpha': 'alpha', r'\beta': 'beta', r'\gamma': 'gamma', r'\delta': 'delta',
        r'\epsilon': 'epsilon', r'\zeta': 'zeta', r'\eta': 'eta', r'\theta': 'theta',
        r'\iota': 'iota', r'\kappa': 'kappa', r'\lambda': 'lambda', r'\mu': 'mu',
        r'\nu': 'nu', r'\xi': 'xi', r'\pi': 'pi', r'\rho': 'rho',
        r'\sigma': 'sigma', r'\tau': 'tau', r'\upsilon': 'upsilon', r'\phi': 'phi',
        r'\chi': 'chi', r'\psi': 'psi', r'\omega': 'omega'
    }
    for latex_greek, hwp_greek in greek_lower.items():
        hwp_eq = hwp_eq.replace(latex_greek, hwp_greek)
    
    # 그리스 문자들 (대문자)
    greek_upper = {
        r'\Alpha': 'ALPHA', r'\Beta': 'BETA', r'\Gamma': 'GAMMA', r'\Delta': 'DELTA',
        r'\Epsilon': 'EPSILON', r'\Zeta': 'ZETA', r'\Eta': 'ETA', r'\Theta': 'THETA',
        r'\Iota': 'IOTA', r'\Kappa': 'KAPPA', r'\Lambda': 'LAMBDA', r'\Mu': 'MU',
        r'\Nu': 'NU', r'\Xi': 'XI', r'\Pi': 'PI', r'\Rho': 'RHO',
        r'\Sigma': 'SIGMA', r'\Tau': 'TAU', r'\Upsilon': 'UPSILON', r'\Phi': 'PHI',
        r'\Chi': 'CHI', r'\Psi': 'PSI', r'\Omega': 'OMEGA'
    }
    for latex_greek, hwp_greek in greek_upper.items():
        hwp_eq = hwp_eq.replace(latex_greek, hwp_greek)
    
    # 관계 연산자
    hwp_eq = hwp_eq.replace(r'\leq', '<=')
    hwp_eq = hwp_eq.replace(r'\geq', '>=')
    hwp_eq = hwp_eq.replace(r'\neq', '!=')
    hwp_eq = hwp_eq.replace(r'\approx', '~~')
    hwp_eq = hwp_eq.replace(r'\equiv', '==')
    
    # 집합 기호
    hwp_eq = hwp_eq.replace(r'\cup', 'union')
    hwp_eq = hwp_eq.replace(r'\cap', 'inter')
    hwp_eq = hwp_eq.replace(r'\subset', 'subset')
    hwp_eq = hwp_eq.replace(r'\supset', 'supset')
    hwp_eq = hwp_eq.replace(r'\in', 'in')
    hwp_eq = hwp_eq.replace(r'\notin', 'notin')
    
    # 박스 기호
    hwp_eq = hwp_eq.replace(r'\Box', 'box{}')
    
    # 점 표시
    hwp_eq = hwp_eq.replace(r'\cdot', 'cdot')
    hwp_eq = hwp_eq.replace(r'\cdots', '...')
    hwp_eq = hwp_eq.replace(r'\ldots', '...')
    hwp_eq = hwp_eq.replace(r'\ddots', 'ddots')
    
    # 텍스트 모드
    hwp_eq = re.sub(r'\\text\{([^}]+)\}', r'"\1"', hwp_eq)
    
    # 연립방정식 (cases): \begin{cases}...\end{cases} -> cases{...}
    # \\ 는 줄바꿈(#)으로 변환
    hwp_eq = re.sub(r'\\begin\{cases\}(.*?)\\end\{cases\}', 
                    lambda m: 'cases{' + re.sub(r'\s*\\\\\s*', '#', m.group(1)).strip() + '}', 
                    hwp_eq, flags=re.DOTALL)
    
    # 행렬: \begin{matrix}...\end{matrix} -> matrix{...}
    hwp_eq = re.sub(r'\\begin\{matrix\}(.*?)\\end\{matrix\}', 
                    lambda m: 'matrix{' + m.group(1).replace(r'\\', ' # ').replace('&', ' & ') + '}', 
                    hwp_eq, flags=re.DOTALL)
    
    hwp_eq = re.sub(r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}', 
                    lambda m: 'pmatrix{' + m.group(1).replace(r'\\', ' # ').replace('&', ' & ') + '}', 
                    hwp_eq, flags=re.DOTALL)
    
    hwp_eq = re.sub(r'\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}', 
                    lambda m: 'bmatrix{' + m.group(1).replace(r'\\', ' # ').replace('&', ' & ') + '}', 
                    hwp_eq, flags=re.DOTALL)
    
    # 남은 백슬래시 제거 (명령어 앞의)
    hwp_eq = hwp_eq.replace('\\', '')
    
    # 공백 정리
    hwp_eq = re.sub(r'\s+', ' ', hwp_eq).strip()
    
    # 긴 수식에 줄바꿈 추가 (RARROW나 = 앞에서 줄바꿈)
    if len(hwp_eq) > max_length:
        # RARROW 앞에 줄바꿈 추가
        hwp_eq = re.sub(r'\s+RARROW\s+', ' # RARROW ', hwp_eq)
        # 등호(=) 앞에 줄바꿈 추가 (단, ==는 제외)
        hwp_eq = re.sub(r'(?<!<)(?<!>)(?<!=)\s+=\s+(?!=)', ' # = ', hwp_eq)
    
    return hwp_eq


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
                hwp_equation = latex_to_hwp_equation(latex)
            except Exception as exc:
                hwp_equation = ""
                print(f"[경고] 한글 수식 변환 실패: {latex} -> {exc}")
            parts.append(("math", latex, hwp_equation))
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
            elif child.name == "b":
                # <b> 태그: 굵은 텍스트 시작/끝 마커 추가
                segments.append(("bold_start", ""))
                segments.extend(collect_segments(child))
                segments.append(("bold_end", ""))
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


def extract_blocks(html_path: Path, page_limit: Union[int, None] = None):
    """본문에서 제목·문단·인용 블록을 순회하며 (태그, 세그먼트 목록)을 구성한다."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    
    # 다양한 HTML 구조 지원: .page-content, .two-column-layout, .a4-page 순으로 찾기
    page_sections = (
        soup.select(".page-content") or 
        soup.select(".two-column-layout") or 
        soup.select(".a4-page") or 
        [soup.body]
    )

    blocks = []
    # 콘텐츠 블록으로 인식할 태그들
    content_tags = ["h1", "h2", "h3", "h4", "p", "li", "blockquote", "div", "span"]
    # 블록 레벨 컨테이너 태그들 (내부를 재귀 탐색)
    container_tags = ["section", "div", "article"]
    
    for page_idx, main in enumerate(page_sections):
        if page_limit is not None and page_idx >= page_limit:
            break
        
        if main is None:
            continue
        
        # 페이지 구분을 위해 빈 줄 추가 (첫 페이지 제외)
        if page_idx > 0:
            blocks.append(("section-br", []))
        
        # 모든 콘텐츠 블록을 순회 (재귀적으로 찾기)
        for node in main.find_all(content_tags, recursive=True):
            # blockquote 내부의 p는 중복이므로 건너뛰기
            if node.name == "p" and node.find_parent("blockquote"):
                continue
            
            # 클래스 기반으로 특수 태그 먼저 확인
            css_classes = node.get('class', []) or []
            
            # 특수 컨테이너 내부의 span은 건너뛰기 (부모에서 처리됨)
            if node.name == "span":
                parent_with_special_class = node.find_parent(
                    class_=['box-header', 'explanation-section', 'answer-box-container', 'unit-title', 'chapter-header']
                )
                if parent_with_special_class:
                    continue
            
            # 특수 클래스를 가진 div는 건너뛰지 않고 처리
            special_classes = ['unit-title', 'chapter-header', 'box-header', 
                              'answer-box-container', 'explanation-section']
            is_special_div = any(cls in css_classes for cls in special_classes)
            
            # div나 span 중 텍스트 콘텐츠가 없는 컨테이너는 건너뛰기 (특수 클래스 제외)
            if node.name in ["div", "span"] and not is_special_div:
                # 자식 중 다른 콘텐츠 태그가 있으면 건너뛰기 (중복 방지)
                if node.find(content_tags):
                    continue
                # 직접 텍스트가 없으면 건너뛰기
                direct_text = ''.join(
                    str(child).strip() for child in node.children 
                    if isinstance(child, NavigableString)
                )
                if not direct_text.strip():
                    continue
            
            segments = collect_segments(node)
            if segments:
                # 클래스 기반으로 특수 태그 처리
                if 'unit-title' in css_classes:
                    blocks.append(("h2", segments))
                elif 'chapter-header' in css_classes:
                    blocks.append(("h1", segments))
                elif 'box-header' in css_classes:
                    blocks.append(("h4", segments))
                elif 'answer-box-container' in css_classes:
                    blocks.append(("blockquote", segments))
                elif 'explanation-section' in css_classes:
                    blocks.append(("p", segments))
                else:
                    blocks.append((node.name, segments))
    
    return blocks


def insert_segments_into_hwp(hwp: Hwp, segments: List[Segment]) -> None:
    """세그먼트를 순회하며 텍스트와 수식을 한글 문서에 삽입한다."""
    for segment in segments:
        kind = segment[0]
        if kind == "text":
            hwp.insert_text(segment[1] + " ")
        elif kind == "break":
            hwp.BreakPara()  # 단순 줄바꿈만 수행
        elif kind == "bold_start":
            hwp.set_font(Bold=True)
        elif kind == "bold_end":
            hwp.set_font(Bold=False)
        elif kind == "math":
            _, latex, hwp_equation = segment
            if not hwp_equation:
                hwp.insert_text(f"[수식 변환 실패: {latex}]")
                continue

            try:
                # 한글 수식 직접 삽입
                hwp.HAction.GetDefault("EquationCreate", hwp.HParameterSet.HEqEdit.HSet)
                
                # 수식 기본 설정
                #hwp.HParameterSet.HEqEdit.EqFontName = "HYhwpEQ"
                hwp.HParameterSet.HEqEdit.EqFontName = "HancomEQN"
                hwp.HParameterSet.HEqEdit.HSet.SetItem("String", hwp_equation)
                hwp.HParameterSet.HEqEdit.BaseUnit = hwp.PointToHwpUnit(11.0)
                
                # 개체 속성: 글자처럼 취급 (TreatAsChar = True)
                hwp.HParameterSet.HEqEdit.TreatAsChar = 1
                
                # 크기 자동 조정: Width와 Height를 0으로 설정하면 내용에 맞춰 자동 조정됨
                hwp.HParameterSet.HEqEdit.Width = 0
                hwp.HParameterSet.HEqEdit.Height = 0
                
                hwp.HAction.Execute("EquationCreate", hwp.HParameterSet.HEqEdit.HSet)
                
                # 수식 선택 해제: ESC 키로 선택 해제 후 오른쪽 화살표로 이동
                hwp.HAction.Run("Escape")
                hwp.Run("MoveLineEnd")  # 라인 끝으로 이동
            except Exception as exc:
                print(f"[경고] 한글 수식 삽입 실패: {latex} -> {hwp_equation} -> {exc}")
                hwp.insert_text(f"[수식 삽입 실패: {latex}]")


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


def create_full_hwpx(blocks, output_path: Path) -> None:
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
        is_paragraph = tag == "p"

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
            # <p> 태그인 경우 추가 빈 줄 삽입
            if is_paragraph:
                hwp.BreakPara()

        if is_heading:
            hwp.set_font(Bold=False)

    hwp.save_as(str(output_path), format="HWPX")
    hwp.quit()
    print(f"[완료] {output_path} 저장")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python convert_unit1_with_hwptex.py <입력HTML파일> [출력HWPX파일]")
        print("예시: python convert_unit1_with_hwptex.py unit1_sample.html")
        print("      python convert_unit1_with_hwptex.py unit1_sample.html output.hwpx")
        sys.exit(1)
    
    # 입력 파일
    html_path = Path(sys.argv[1])
    if not html_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {html_path}")
        sys.exit(1)
    
    # 출력 파일 (지정하지 않으면 입력파일명_hwptex.hwpx)
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2]).resolve()
    else:
        output_path = html_path.with_name(html_path.stem + "_hwptex.hwpx").resolve()
    
    print(f"[시작] HTML 파일: {html_path}")
    print(f"[시작] 출력 파일: {output_path}")
    
    blocks = extract_blocks(html_path, page_limit=None) 
    create_full_hwpx(blocks, output_path)
