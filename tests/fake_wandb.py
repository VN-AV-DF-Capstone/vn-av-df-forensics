"""Module `wandb` giả cho test: ghi lại mọi run, key đã log, video, artifact (không cần mạng)."""

import itertools


class Table:
    def __init__(self, columns, data):
        self.columns, self.data = columns, data


class Histogram:
    def __init__(self, values):
        self.values = list(values)


class Video:
    def __init__(self, path, caption=None, format=None):
        self.path, self.caption = path, caption


class Plot:
    def line(self, table, x, y, stroke=None, title=None):
        return ("line", table, x, y, stroke)

    def bar(self, table, label, value, title=None):
        return ("bar", table, label, value)


class Artifact:
    def __init__(self, name, type, metadata=None):
        self.name, self.type, self.metadata, self.files = name, type, metadata, []

    def add_file(self, path):
        self.files.append(path)

    def add_dir(self, path, name=None):
        self.files.append(path)


class FakeRun:
    ids = itertools.count(1)

    def __init__(self, store, **kwargs):
        self.store, self.kwargs = store, kwargs
        self.id = kwargs.get("id") or f"run{next(self.ids)}"
        self.logged, self.summary, self.artifacts, self.finished = [], {}, [], False
        store.runs.append(self)

    def log(self, data, step=None):
        self.logged.append(dict(data))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self):
        self.finished = True

    def keys(self):
        return {k for entry in self.logged for k in entry}


class FakeWandb:
    Table, Histogram, Video, Artifact = Table, Histogram, Video, Artifact

    def __init__(self):
        self.runs = []
        self.plot = Plot()

    def init(self, **kwargs):
        return FakeRun(self, **kwargs)

    def by_type(self, job_type):
        return [r for r in self.runs if r.kwargs.get("job_type") == job_type]
