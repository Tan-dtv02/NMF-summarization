"""Các hàm vẽ dùng chung cho ba màn: màu chủ đề, bảng ma trận, biểu đồ.

Tệp này chỉ lo phần hiển thị. Không có thuật toán nào ở đây.
"""

from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd

# Màu của các chủ đề, dùng theo đúng thứ tự này. Bộ màu đã được kiểm tra để
# người mù màu vẫn phân biệt được hai màu đứng cạnh nhau. Chủ đề luôn kèm nhãn
# chữ "CĐ 1", "CĐ 2"... nên không ai phải dựa vào màu để nhận ra chủ đề.
TOPIC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

INK = "#0b0b0b"  # chữ chính
TOPIC_INK = INK  # chữ trên nền màu chủ đề: luôn sẫm, vì nền luôn nhạt dù giao diện sáng hay tối
INK_SOFT = "#52514e"  # chữ phụ
GRID = "#e4e3df"  # đường lưới, mờ hơn dữ liệu
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
UP_COLOR = "#cde2fb"  # ô tăng so với trước
DOWN_COLOR = "#fbd9cc"  # ô giảm so với trước

FONT_SIZE = 15  # cỡ chữ trong biểu đồ, đủ lớn để nhìn qua máy chiếu


def topic_color(topic: int) -> str:
    return TOPIC_COLORS[topic % len(TOPIC_COLORS)]


def tint(hex_color: str, amount: float = 0.8) -> str:
    """Pha màu với trắng. amount = 0.8 nghĩa là 80% trắng, dùng làm nền chữ."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def topic_chip(topic: int) -> str:
    """Nhãn nhỏ có màu: "CĐ 1", "CĐ 2"... (HTML)."""
    if topic < 0:
        return '<span class="chip" style="background:#eeeeec">–</span>'
    return (
        f'<span class="chip" style="background:{tint(topic_color(topic), 0.8)}">'
        f'<i style="background:{topic_color(topic)}"></i>CĐ {topic + 1}</span>'
    )


def matrix_html(
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    previous: np.ndarray | None = None,
    color_by: str = "value",
    decimals: int = 3,
) -> str:
    """Vẽ một ma trận thành bảng HTML chữ lớn.

    color_by = "value":  ô càng lớn thì nền xanh càng đậm (đọc cấu trúc).
    color_by = "change": ô tăng so với `previous` có nền xanh và mũi tên lên,
                         ô giảm có nền cam và mũi tên xuống (đọc từng bước).
    """
    top = float(values.max()) if values.size and values.max() > 0 else 1.0
    head = "".join(f"<th>{html.escape(c)}</th>" for c in col_labels)
    body = []
    for i, label in enumerate(row_labels):
        cells = []
        for j in range(values.shape[1]):
            value = float(values[i, j])
            background, mark = "#ffffff", ""
            if color_by == "change" and previous is not None:
                delta = round(value, decimals) - round(float(previous[i, j]), decimals)
                if delta > 0:
                    background, mark = UP_COLOR, "▲"
                elif delta < 0:
                    background, mark = DOWN_COLOR, "▼"
            elif color_by == "value":
                background = tint(BLUE, 1 - 0.6 * value / top)
            # Cột mũi tên chỉ cần khi đang xem thay đổi; bỏ đi thì bảng gọn hơn.
            arrow = f'<span class="mark">{mark}</span>' if color_by == "change" else ""
            cells.append(f'<td style="background:{background}">{value:.{decimals}f}{arrow}</td>')
        body.append(f"<tr><th>{html.escape(label)}</th>{''.join(cells)}</tr>")
    return (
        f'<div class="matrix-wrap"><table class="matrix"><thead><tr><th></th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _styled(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    """Kiểu chung cho mọi biểu đồ: chữ lớn, lưới mờ, không viền khung."""
    return (
        chart.configure_axis(
            labelFontSize=FONT_SIZE,
            titleFontSize=FONT_SIZE,
            labelColor=INK_SOFT,
            titleColor=INK_SOFT,
            titleFontWeight="normal",
            gridColor=GRID,
            domainColor=GRID,
            tickColor=GRID,
            labelLimit=260,
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelFontSize=FONT_SIZE, titleFontSize=FONT_SIZE, labelColor=INK)
        .configure_header(labelFontSize=FONT_SIZE + 1, labelColor=INK, titleFontSize=FONT_SIZE)
    )


def error_chart(errors: list[float], height: int = 260) -> alt.LayerChart:
    """Đường sai số theo số vòng lặp."""
    data = pd.DataFrame({"Vòng lặp": range(len(errors)), "Sai số": errors})
    # Ở vòng 0 mới có một điểm: tự đặt khoảng của hai trục, nếu không thư viện vẽ trục
    # chỉ có đúng một vạch và điểm nằm lơ lửng giữa ô.
    one_point = len(errors) == 1
    x_scale = alt.Scale(domain=[0, 1]) if one_point else alt.Scale()
    y_scale = alt.Scale(domain=[0, errors[0] * 1.15]) if one_point else alt.Scale(zero=False)
    base = alt.Chart(data).encode(
        x=alt.X("Vòng lặp:Q", scale=x_scale, axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y("Sai số:Q", scale=y_scale, title="Sai số ||V − W·H||"),
        tooltip=["Vòng lặp:Q", alt.Tooltip("Sai số:Q", format=".4f")],
    )
    line = base.mark_line(color=BLUE, strokeWidth=2)
    points = base.mark_circle(color=BLUE, size=70, opacity=1, stroke="white", strokeWidth=2)
    return _styled((line + points).properties(height=height))


def topic_map_chart(H: np.ndarray, sentences: list[str], selected: list[int]) -> alt.LayerChart:
    """Bản đồ chủ đề: mỗi cột là một câu (theo thứ tự trong bài), mỗi hàng là
    một chủ đề, ô càng đậm thì câu càng thuộc chủ đề đó. Câu được chọn vào bản
    tóm tắt có dấu ★ phía trên."""
    k, n = H.shape
    share = H / np.maximum(H.sum(axis=0, keepdims=True), 1e-12)  # tỉ lệ chủ đề trong từng câu
    rows = [
        {
            "Câu": j + 1,
            "Chủ đề": f"CĐ {t + 1}",
            "Tỉ lệ": float(share[t, j]),
            "Nội dung": sentences[j],
        }
        for t in range(k)
        for j in range(n)
    ]
    x = alt.X("Câu:O", title="Câu (theo thứ tự trong bài)", axis=alt.Axis(labelAngle=0))
    cells = (
        alt.Chart(pd.DataFrame(rows))
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("Chủ đề:N", title=None),
            color=alt.Color(
                "Tỉ lệ:Q",
                scale=alt.Scale(range=["#eef4fd", "#0d366b"], domain=[0, 1]),
                legend=alt.Legend(title="Tỉ lệ chủ đề trong câu", format=".0%"),
            ),
            tooltip=[
                "Câu:O",
                "Chủ đề:N",
                alt.Tooltip("Tỉ lệ:Q", format=".0%"),
                "Nội dung:N",
            ],
        )
    )
    stars = (
        alt.Chart(pd.DataFrame({"Câu": [j + 1 for j in selected], "Dấu": "★"}))
        .mark_text(fontSize=22, color=INK, baseline="bottom", dy=-4)
        .encode(x=x, y=alt.value(0), text="Dấu:N")
    )
    # Chiều cao tính cả trục và hàng dấu ★, nên cộng thêm phần lề ngoài k hàng ô.
    return _styled((cells + stars).properties(height=56 * k + 100))


def is_ceiling(method: str) -> bool:
    """Dòng "mức trần" trong bảng điểm: chọn câu khi đã biết bản tham chiếu."""
    return "mức trần" in str(method)


# Mỗi nhóm phương pháp luôn mang một màu cố định, dù tệp kết quả có nhóm đó hay không.
METHOD_GROUP_COLORS = {"NMF": BLUE, "Đối chứng": ORANGE, "Mức trần": GREEN}


def method_group(method: str) -> str:
    """Nhóm của một phương pháp trong bảng điểm: NMF, Đối chứng hay Mức trần."""
    if str(method).startswith("NMF"):
        return "NMF"
    return "Mức trần" if is_ceiling(method) else "Đối chứng"


def rouge_chart(table: pd.DataFrame) -> alt.FacetChart:
    """Điểm F1 của từng phương pháp: ba ô cạnh nhau cho ROUGE-1, ROUGE-2, ROUGE-L."""
    names = {"rouge1_f": "ROUGE-1", "rouge2_f": "ROUGE-2", "rougeL_f": "ROUGE-L"}
    long = table.melt(
        id_vars="phuong_phap", value_vars=list(names), var_name="Độ đo", value_name="F1"
    )
    long["Độ đo"] = long["Độ đo"].map(names)
    long["Nhóm"] = long["phuong_phap"].map(method_group)
    order = list(table["phuong_phap"])
    groups = [group for group in METHOD_GROUP_COLORS if group in set(long["Nhóm"])]

    base = alt.Chart().encode(
        y=alt.Y("phuong_phap:N", sort=order, title=None),
        x=alt.X("F1:Q", title="F1", axis=alt.Axis(format=".2f", tickCount=4)),
    )
    bars = base.mark_bar(size=20, cornerRadiusEnd=5).encode(
        color=alt.Color(
            "Nhóm:N",
            scale=alt.Scale(domain=groups, range=[METHOD_GROUP_COLORS[g] for g in groups]),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=["phuong_phap:N", "Độ đo:N", alt.Tooltip("F1:Q", format=".4f")],
    )
    values = base.mark_text(align="left", dx=5, fontSize=FONT_SIZE - 1, color=INK).encode(
        text=alt.Text("F1:Q", format=".3f")
    )
    chart = (
        alt.layer(bars, values, data=long)
        .properties(width=250, height=38 * len(order))
        .facet(column=alt.Column("Độ đo:N", title=None, sort=list(names.values())))
    )
    return _styled(chart)
