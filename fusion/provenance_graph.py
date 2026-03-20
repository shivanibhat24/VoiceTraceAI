"""
Layer 4 — Provenance Graph Builder
Creates a NetworkX graph linking: fake audio → cloning tool → source speaker → source URL.
Visualized as an interactive HTML graph using pyvis.
"""

import networkx as nx
from pathlib import Path
from loguru import logger

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    logger.warning("pyvis not installed. HTML graph export disabled.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


TOOL_COLORS = {
    "real":        "#1D9E75",
    "elevenlabs":  "#A32D2D",
    "coqui":       "#BA7517",
    "rvc":         "#534AB7",
    "openvoice":   "#0F6E56",
    "unknown":     "#888780",
}


def build_provenance_graph(result) -> nx.DiGraph:
    """
    Build a directed provenance graph from an AttributionResult.
    Nodes: audio clip, cloning tool, source speaker(s), evidence node.
    Edges: attributed_to, cloned_from, evidence.
    """
    G = nx.DiGraph()

    # Central node: the submitted audio
    audio_id = "submitted_audio"
    G.add_node(audio_id, label="Submitted Audio",
               node_type="audio", color="#2C2C2A",
               size=30, shape="ellipse")

    # Tool node
    tool = result.tool_label
    tool_id = f"tool_{tool}"
    tool_color = TOOL_COLORS.get(tool, TOOL_COLORS["unknown"])
    G.add_node(tool_id,
               label=f"Tool: {tool.upper()}",
               node_type="tool",
               color=tool_color,
               size=25,
               confidence=result.tool_confidence)

    G.add_edge(audio_id, tool_id,
               label=f"attributed ({result.tool_confidence:.0%})",
               weight=result.tool_confidence,
               color=tool_color)

    # Speaker nodes
    for sp in result.top_speakers:
        sp_id = f"speaker_{sp['speaker_id']}"
        alpha = int(sp["similarity"] * 200 + 55)  # 55–255
        G.add_node(sp_id,
                   label=f"Speaker: {sp['speaker_id']}",
                   node_type="speaker",
                   color="#534AB7",
                   size=20,
                   rank=sp["rank"],
                   similarity=sp["similarity"])

        G.add_edge(tool_id, sp_id,
                   label=f"rank {sp['rank']} ({sp['similarity']:.0%})",
                   weight=sp["similarity"],
                   color="#534AB7")

    # Verdict node
    verdict_id = "verdict"
    verdict_color = "#A32D2D" if result.is_fake else "#1D9E75"
    verdict_label = f"{'FAKE' if result.is_fake else 'REAL'}\n{result.fusion_score:.0%}"
    G.add_node(verdict_id,
               label=verdict_label,
               node_type="verdict",
               color=verdict_color,
               size=35)

    G.add_edge(audio_id, verdict_id,
               label=f"fusion ({result.fusion_score:.0%})",
               weight=result.fusion_score,
               color=verdict_color)

    return G


def export_html(G: nx.DiGraph, out_path: str,
                height: str = "500px", width: str = "100%") -> str:
    """Export provenance graph as interactive HTML using pyvis."""
    if not PYVIS_AVAILABLE:
        logger.warning("pyvis not available — skipping HTML export")
        return ""

    net = Network(height=height, width=width, directed=True,
                  bgcolor="#FFFFFF", font_color="#2C2C2A")
    net.options.physics.enabled = True

    for node_id, attrs in G.nodes(data=True):
        net.add_node(
            node_id,
            label=attrs.get("label", node_id),
            color=attrs.get("color", "#888780"),
            size=attrs.get("size", 20),
            shape=attrs.get("shape", "dot"),
            title=f"Type: {attrs.get('node_type', '')}\n"
                  f"Confidence: {attrs.get('confidence', attrs.get('similarity', '')):.0%}"
                  if isinstance(attrs.get("confidence", attrs.get("similarity")), float)
                  else attrs.get("label", ""),
        )

    for u, v, attrs in G.edges(data=True):
        net.add_edge(
            u, v,
            label=attrs.get("label", ""),
            color=attrs.get("color", "#888780"),
            width=max(1, attrs.get("weight", 0.5) * 4),
            arrows="to",
        )

    out_path = str(out_path)
    net.save_graph(out_path)
    logger.success(f"Provenance graph saved → {out_path}")
    return out_path


def export_matplotlib(G: nx.DiGraph, out_path: str) -> str:
    """Export provenance graph as a static matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#F8F8F6")
    fig.patch.set_facecolor("#F8F8F6")

    pos = nx.spring_layout(G, seed=42, k=2)

    node_colors = [G.nodes[n].get("color", "#888780") for n in G.nodes]
    node_sizes = [G.nodes[n].get("size", 20) * 80 for n in G.nodes]
    node_labels = {n: G.nodes[n].get("label", n) for n in G.nodes}

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                            node_size=node_sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8,
                             font_color="#1A1A18", ax=ax)

    edge_colors = [G.edges[e].get("color", "#888780") for e in G.edges]
    edge_weights = [max(1, G.edges[e].get("weight", 0.5) * 3) for e in G.edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights,
                            arrows=True, arrowsize=15, ax=ax,
                            connectionstyle="arc3,rad=0.1")

    edge_labels = {(u, v): G.edges[u, v].get("label", "") for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=7, font_color="#555550", ax=ax)

    # Legend
    legend_patches = [
        mpatches.Patch(color=TOOL_COLORS[t], label=t.upper())
        for t in ["real", "elevenlabs", "coqui", "rvc", "openvoice"]
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("VoiceTrace — Attribution Provenance Graph", fontsize=13,
                  fontweight="bold", color="#2C2C2A")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Provenance graph (matplotlib) saved → {out_path}")
    return out_path


def generate_provenance_outputs(result, out_dir: str) -> dict:
    """
    Generate both HTML and PNG provenance graphs from an AttributionResult.
    Returns dict of {html_path, png_path}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = build_provenance_graph(result)

    outputs = {}
    png_path = str(out_dir / "provenance_graph.png")
    outputs["png"] = export_matplotlib(G, png_path)

    if PYVIS_AVAILABLE:
        html_path = str(out_dir / "provenance_graph.html")
        outputs["html"] = export_html(G, html_path)

    return outputs
