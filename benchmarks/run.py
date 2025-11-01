
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SOLUTION_DIR = Path("solutions")
BEST_SOLUTION_FILE = Path("best_solution.txt")


def draw(op: str, data: dict[str, float], result_dir: Path):
    print("op:", op)
    print("data:", data)

    tool_names = list(data.keys())
    times = list(data.values())

    num_tools = len(tool_names)
    fig_width = max(8, min(12, num_tools * 1.2))
    plt.figure(figsize=(fig_width, 6))

    bar_width = 0.7
    x_pos = np.arange(len(tool_names))

    colors = ['#e74c3c' if name ==
              "best" else '#3498db' for name in tool_names]

    bars = plt.bar(x_pos, times, width=bar_width,
                   color=colors,
                   edgecolor='white', linewidth=1.5, alpha=0.9)

    for bar, time_val in zip(bars, times):
        if time_val < max(times) * 0.3:
            label_y = bar.get_height() - max(times) * 0.03
            va = 'top'
            color = 'white'
        else:
            label_y = bar.get_height() + max(times) * 0.01
            va = 'bottom'
            color = 'black'

        plt.text(bar.get_x() + bar.get_width()/2, label_y,
                 f'{time_val:.2f}', ha='center', va=va,
                 fontsize=10, fontweight='bold', color=color)

    plt.xlabel('Implementation', fontsize=12, fontweight='bold')
    plt.ylabel('Time Cost (ms)', fontsize=12, fontweight='bold')
    plt.title(f'Performance Comparison for {op}',
              fontsize=14, fontweight='bold', pad=20)

    # Baseline
    if "best" in data:
        plt.axhline(y=data["best"], color='#e74c3c', linestyle='--',
                    alpha=0.7, linewidth=2, label='Best Solution Baseline')

    # x-axis label
    plt.xticks(x_pos, tool_names, rotation=45, ha='right')

    # Grid
    plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    if "best" in data:
        plt.legend(frameon=True, fancybox=True, shadow=True)

    # y-axis
    y_max = max(times) * 1.15
    plt.ylim(0, y_max)

    # Color
    plt.gca().set_facecolor('#f8f9fa')
    plt.gcf().patch.set_facecolor('white')
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#cccccc')

    plt.tight_layout()

    # Save
    save_path = result_dir / f"{op}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Chart saved to: {save_path}")


def get_best_solution(filepath: Path) -> float:
    with filepath.open("r") as f:
        text = f.read()
        time = text.strip().split(' ')[1]
        return float(time[:-2])
    return 0.0


def get_time_from_solution(filepath: Path) -> float:
    with filepath.open("r") as f:
        text = f.readline()
        time = text.strip().split(' ')[1]
        return float(time[:-2])
    return 0.0


def get_time_data(op: str) -> dict[str, float]:
    data = {}
    op_dir = SOLUTION_DIR / op
    # Best solution
    best = get_best_solution(op_dir / BEST_SOLUTION_FILE)
    data["best"] = best
    # Other solution
    for tool in op_dir.iterdir():
        if tool.is_dir():
            tool_name = tool.stem
            pattern = f"{op}_{tool_name}.*"
            solution_files = list(tool.glob(pattern))
            time = get_time_from_solution(solution_files[0])
            data[tool_name] = time
    return data


def get_all_ops() -> list[str]:
    return [op_dir.stem for op_dir in SOLUTION_DIR.iterdir()]


def create_result_dir():
    result_dir = Path("benchmarks") / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def main():
    result_dir = create_result_dir()
    ops = get_all_ops()
    print("ops:", ops)
    for op in ops:
        time_data = get_time_data(op)
        draw(op, time_data, result_dir)


if __name__ == "__main__":
    main()
