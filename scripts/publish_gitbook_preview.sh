#!/usr/bin/env bash
set -euo pipefail

book=$(realpath "$1")
git_dir=$(git rev-parse --absolute-git-dir)
source=$(git rev-parse HEAD)
ref=refs/heads/gitbook-preview
[[ "$(<"$book/SOURCE_REVISION")" == "$source" ]] || {
    echo 'The GitBook tree was built from a different source revision.' >&2
    exit 1
}

parent=()
if git ls-remote --exit-code --heads origin "$ref" >/dev/null; then
    git fetch --no-tags origin "$ref"
    previous=$(git rev-parse FETCH_HEAD)
    previous_source=$(git show "$previous:SOURCE_REVISION")
    [[ "$previous_source" =~ ^[0-9a-f]{40}$ &&
       "$(git show -s --format=%s "$previous")" == "Publish GitBook preview from $previous_source" ]] || {
        echo 'Review the existing GitBook branch before replacing its content.' >&2
        exit 1
    }
    if git merge-base --is-ancestor "$source" "$previous_source"; then
        echo "GitBook preview already contains $previous_source."
        exit 0
    fi
    git merge-base --is-ancestor "$previous_source" "$source" || {
        echo 'The existing GitBook branch records unrelated source history.' >&2
        exit 1
    }
    parent=(-p "$previous")
else
    result=$?
    [[ "$result" == 2 ]] || exit "$result"
fi

scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
export GIT_INDEX_FILE="$scratch/preview-index"
git -C "$book" --git-dir="$git_dir" --work-tree="$book" add -f -- .
tree=$(git write-tree)
preview=$(git -c user.name='github-actions[bot]' \
    -c user.email='41898282+github-actions[bot]@users.noreply.github.com' \
    -c commit.gpgsign=false commit-tree "$tree" "${parent[@]}" -m "Publish GitBook preview from $source")
git push origin "$preview:$ref"
[[ "$(git ls-remote --heads origin "$ref")" == "$preview"$'\t'"$ref" ]]
echo "Published GitBook preview $preview from $source."
