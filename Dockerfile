# eda-rl + real OpenROAD-flow-scripts tools, for nightly F2/F3 builds and
# tagged GHCR releases. Layers eda-rl on top of the OpenROAD project's own
# prebuilt ORFS image rather than building OpenROAD/yosys from source here
# (a from-source build takes hours and doesn't belong in CI).
#
# Tag scheme note: openroad/orfs tags are <quarter>-<commit-count>-g<hash>,
# not semver — there is no automated bump path, this needs periodic manual
# updates. Check https://hub.docker.com/r/openroad/orfs/tags for current tags.
FROM openroad/orfs:26Q1-578-g0a7c9ffc8

WORKDIR /eda-rl
COPY . .

# The pinned base image's pip is old enough to predate PEP 660 (editable
# installs from a pyproject.toml-only project need pip >=21.3) — confirmed
# empirically: it fails with "build backend is missing the build_editable
# hook". Upgrade pip itself first rather than working around that version
# gap. Editable install is deliberate, not incidental: pyproject.toml's
# package-data only globs **/*.yaml/**/*.yml, so a non-editable
# `pip install .` would silently drop vendored RTL (eda_rl/designs/gcd/gcd.v
# etc.) from the installed package. Fall back to --break-system-packages in
# case the upgraded pip lands on a distro that enforces PEP 668.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    (python3 -m pip install --no-cache-dir -e . || \
     python3 -m pip install --no-cache-dir --break-system-packages -e .)

ENV ORFS_DIR=/OpenROAD-flow-scripts
ENV EDA_RL_WORK=/work
