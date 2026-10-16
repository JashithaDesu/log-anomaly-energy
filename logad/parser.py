"""A small implementation of Drain (He et al., ICWS 2017), the standard online
log parser: it turns raw log lines into event templates such as
    "PacketResponder <*> for block <*> terminating"
so that a log stream becomes a sequence of event IDs a model can learn from.

Drain groups lines with a fixed-depth tree: first by token count, then by the
first few tokens, then by similarity to existing templates. It needs one pass
and no training, which makes it cheap enough to run on every log line.
"""
import re

# Variables that should never become part of a template (applied before parsing)
MASKS = [
    (re.compile(r"blk_-?\d+"), "<*>"),                         # HDFS block IDs
    (re.compile(r"(\d+\.){3}\d+(:\d+)?"), "<*>"),              # IP addresses (and ports)
    (re.compile(r"(?<![A-Za-z])-?\d+(\.\d+)?(?![A-Za-z])"), "<*>"),  # plain numbers
]


def preprocess(content):
    for rx, rep in MASKS:
        content = rx.sub(rep, content)
    return content.split()


class Drain:
    def __init__(self, depth=4, sim_threshold=0.5, max_children=100):
        self.depth = depth - 2            # tree levels used for leading tokens
        self.st = sim_threshold
        self.max_children = max_children
        self.root = {}
        self.templates = []               # template id -> list of tokens

    @staticmethod
    def _sim(template, tokens):
        same = sum(1 for a, b in zip(template, tokens) if a == b and a != "<*>")
        params = sum(1 for a in template if a == "<*>")
        return same / len(tokens), params

    def _leaf(self, tokens):
        node = self.root.setdefault(len(tokens), {})
        for tok in tokens[: self.depth]:
            key = "<*>" if any(c.isdigit() for c in tok) else tok
            if key not in node:
                key = key if len(node) < self.max_children else "<*>"
            node = node.setdefault(key, {})
        return node.setdefault("_clusters", [])

    def add(self, content):
        """Parse one log message; returns its template id."""
        tokens = preprocess(content)
        if not tokens:
            tokens = ["<empty>"]
        clusters = self._leaf(tokens)
        best, best_sim, best_params = None, -1.0, -1
        for tid in clusters:
            sim, params = self._sim(self.templates[tid], tokens)
            if sim > best_sim or (sim == best_sim and params > best_params):
                best, best_sim, best_params = tid, sim, params
        if best is not None and best_sim >= self.st:
            t = self.templates[best]
            self.templates[best] = [a if a == b else "<*>" for a, b in zip(t, tokens)]
            return best
        self.templates.append(tokens)
        clusters.append(len(self.templates) - 1)
        return len(self.templates) - 1

    def template(self, tid):
        return " ".join(self.templates[tid])


HDFS_LINE = re.compile(r"^(\d{6}) (\d{6}) (\d+) (\w+) ([^:]+): (.*)$")


def hdfs_content(line):
    """Message part of an HDFS log line (after 'date time pid level component:')."""
    m = HDFS_LINE.match(line.rstrip("\n"))
    return m.group(6) if m else line.strip()


def grouping_accuracy(predicted, truth):
    """Standard log-parsing metric (Zhu et al., ICSE-SEIP 2019): share of lines
    whose predicted group contains exactly the same lines as their true group."""
    from collections import defaultdict
    pg, tg = defaultdict(set), defaultdict(set)
    for i, (p, t) in enumerate(zip(predicted, truth)):
        pg[p].add(i)
        tg[t].add(i)
    correct = sum(len(lines) for t, lines in tg.items()
                  if any(pg[predicted[i]] == lines for i in list(lines)[:1]))
    return correct / len(truth)
