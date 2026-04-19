#!/bin/bash
# install.sh — bootstrap the r3LAY Hermes profile
#
# Sets up ~/.hermes/profiles/r3lay/ as a partial-bind profile:
#   - SOUL.md and skills/ are SYMLINKED to the repo (auto-update on git pull)
#   - config.yaml and .env are REAL files (user-local, never overwritten)
#
# Idempotent — re-running preserves user-edited config.yaml and .env.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$REPO_DIR/hermes-profile"
PROFILE_DIR="$HOME/.hermes/profiles/r3lay"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: $SOURCE_DIR not found. Are you running install.sh from the r3LAY repo root?" >&2
    exit 1
fi

# Tear down any existing whole-directory symlink (legacy install)
# Back up instead of deleting — protects against accidental data loss on a
# user-customized install.
if [ -L "$PROFILE_DIR" ]; then
    backup="${PROFILE_DIR}.bak.$(date +%s)"
    echo "Moving legacy symlink aside: $PROFILE_DIR -> $backup"
    mv "$PROFILE_DIR" "$backup"
fi

mkdir -p "$PROFILE_DIR"

# Symlink code-like artifacts (auto-update with `git pull`)
ln -sfn "$SOURCE_DIR/SOUL.md" "$PROFILE_DIR/SOUL.md"
ln -sfn "$SOURCE_DIR/skills" "$PROFILE_DIR/skills"
echo "Linked: SOUL.md, skills/"

# Copy template only if no real config exists (preserves user edits).
# The template uses __HOME__ and __REPO_DIR__ placeholders which we
# substitute here — Hermes's YAML loader does NOT do env var expansion,
# so path templating has to happen at install time.
if [ ! -f "$PROFILE_DIR/config.yaml" ]; then
    # Use a temp file then atomic rename so a partial write can't leave
    # a corrupt config in place.
    #
    # We use python3 for the substitution instead of sed because sed's
    # replacement text has special characters (&, |, \) that would be
    # misinterpreted if $HOME or $REPO_DIR happen to contain them. A
    # Python str.replace() has no metacharacters.
    tmp="$(mktemp)"
    python3 -c "
import sys
text = open(sys.argv[1]).read()
text = text.replace('__HOME__', sys.argv[2])
text = text.replace('__REPO_DIR__', sys.argv[3])
open(sys.argv[4], 'w').write(text)
" "$SOURCE_DIR/config.template.yaml" "$HOME" "$REPO_DIR" "$tmp"
    mv "$tmp" "$PROFILE_DIR/config.yaml"
    echo "Created: $PROFILE_DIR/config.yaml (rendered from template)"
    echo "         Paths substituted: HOME=$HOME REPO_DIR=$REPO_DIR"
else
    echo "Kept existing: $PROFILE_DIR/config.yaml"
    echo "         (template not re-rendered — edit manually if HOME changed)"
fi

if [ ! -f "$PROFILE_DIR/.env" ]; then
    cp "$SOURCE_DIR/.env.template" "$PROFILE_DIR/.env"
    echo "Created: $PROFILE_DIR/.env (from template — add your OPENROUTER_API_KEY)"
else
    echo "Kept existing: $PROFILE_DIR/.env"
fi

# Also copy r3LAY backend config template if missing
if [ ! -f "$REPO_DIR/r3lay-config.yaml" ]; then
    cp "$REPO_DIR/r3lay-config.template.yaml" "$REPO_DIR/r3lay-config.yaml"
    echo "Created: $REPO_DIR/r3lay-config.yaml (from template — edit to customize)"
fi

# Install the r3LAY skin into the PROFILE's skins dir (always update — skin
# is code, not user config). Each Hermes profile has its own HERMES_HOME, so
# skins must live at ~/.hermes/profiles/r3lay/skins/, NOT ~/.hermes/skins/.
mkdir -p "$PROFILE_DIR/skins"
cp "$SOURCE_DIR/skins/r3lay.yaml" "$PROFILE_DIR/skins/r3lay.yaml"
echo "Installed skin: $PROFILE_DIR/skins/r3lay.yaml"

# ─────────────────────────────────────────────────────────
# Shell-agnostic `r3` launcher via /usr/local/bin symlink
# ─────────────────────────────────────────────────────────
# Without this, users need to add r3lay/ to their shell PATH manually
# (different syntax per shell: fish_user_paths, bash PATH export, zsh
# path array). A symlink in /usr/local/bin works from every shell with
# zero config and survives repo updates.
#
# Requires sudo. Skipped if the user declines or sudo isn't available.
R3_SYMLINK="/usr/local/bin/r3"
R3_SCRIPT="$REPO_DIR/r3lay/r3"

if [ -L "$R3_SYMLINK" ]; then
    # Symlink already exists — check if it points at this install or
    # at an older location, and offer to update if stale.
    current_target="$(readlink "$R3_SYMLINK")"
    if [ "$current_target" = "$R3_SCRIPT" ]; then
        echo "Symlink already current: $R3_SYMLINK -> $R3_SCRIPT"
    else
        echo ""
        echo "Note: $R3_SYMLINK points at a different install:"
        echo "      current: $current_target"
        echo "      this:    $R3_SCRIPT"
        read -r -p "Update the symlink to this install? [y/N] " response
        if [[ "$response" =~ ^[Yy] ]]; then
            sudo ln -sfn "$R3_SCRIPT" "$R3_SYMLINK" && \
                echo "Updated: $R3_SYMLINK -> $R3_SCRIPT"
        fi
    fi
elif [ -e "$R3_SYMLINK" ]; then
    echo "Warning: $R3_SYMLINK exists but is not a symlink. Skipping."
    echo "         If you want the shell-agnostic launcher, remove it first:"
    echo "           sudo rm $R3_SYMLINK && ./install.sh"
else
    echo ""
    echo "Optional: create a shell-agnostic 'r3' launcher at $R3_SYMLINK"
    echo "          so you can run 'r3 up' from any shell without PATH setup."
    echo "          (Requires sudo — you'll be prompted for your password.)"
    read -r -p "Create $R3_SYMLINK? [Y/n] " response
    if [[ ! "$response" =~ ^[Nn] ]]; then
        if sudo ln -sfn "$R3_SCRIPT" "$R3_SYMLINK"; then
            echo "Created: $R3_SYMLINK -> $R3_SCRIPT"
        else
            echo "Skipped: could not create symlink (sudo failed or declined)"
            echo "         You can add r3lay/ to your shell PATH instead:"
            echo "           fish:  set -U fish_user_paths $REPO_DIR/r3lay \$fish_user_paths"
            echo "           bash:  echo 'export PATH=\"$REPO_DIR/r3lay:\$PATH\"' >> ~/.bashrc"
            echo "           zsh:   echo 'export PATH=\"$REPO_DIR/r3lay:\$PATH\"' >> ~/.zshrc"
        fi
    else
        echo "Skipped symlink. Add r3lay/ to your shell PATH manually:"
        echo "  fish:  set -U fish_user_paths $REPO_DIR/r3lay \$fish_user_paths"
        echo "  bash:  echo 'export PATH=\"$REPO_DIR/r3lay:\$PATH\"' >> ~/.bashrc"
        echo "  zsh:   echo 'export PATH=\"$REPO_DIR/r3lay:\$PATH\"' >> ~/.zshrc"
    fi
fi

echo ""
echo "Hermes profile installed at: $PROFILE_DIR"
echo ""
echo "Next steps:"
echo "  1. Edit $PROFILE_DIR/.env and add your OPENROUTER_API_KEY"
echo "     (get a free key at https://openrouter.ai/keys)"
echo "  2. Edit $REPO_DIR/r3lay-config.yaml if you want to customize the backend"
echo "  3. Start the backend:  r3 up"
echo "  4. Start the agent:    r3"
