import os
import sys
from pathlib import Path

import click
import platformdirs

from .build import GalleryError, build_gallery


def _progress(done: int, total: int) -> None:
    if sys.stderr.isatty():
        click.echo(f"\rAnalysing {done}/{total}", nl=done == total, err=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("src", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--title", "-t", required=True, help="Page title; also names the output files.")
@click.option(
    "--count",
    "-n",
    default=48,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of images to show.",
)
@click.option(
    "--output-dir",
    "-o",
    default=Path("."),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Analysis cache (default: the user cache directory).",
)
@click.option("--no-cache", is_flag=True, help="Do not read or write the analysis cache.")
@click.option(
    "--jobs",
    "-j",
    type=click.IntRange(min=1),
    default=None,
    help="Parallel workers (default: CPU count).",
)
@click.option(
    "--order",
    type=click.Choice(["chronological", "varied"]),
    default="chronological",
    show_default=True,
    help="chronological: by capture time; varied: neighbours look as different as possible.",
)
@click.option(
    "--event-quota/--no-event-quota",
    default=False,
    show_default=True,
    help="Give every event (a run of days with photos) at least one image, "
    "so small events are not drowned out by large ones.",
)
@click.option(
    "--balance",
    type=click.FloatRange(0, 1),
    default=1.0,
    show_default=True,
    help="Down-weight images in crowded regions: 0 treats every image alike, "
    "1 gives sparse regions the same weight as dense ones.",
)
def main(
    src: Path,
    title: str,
    count: int,
    output_dir: Path,
    cache_dir: Path | None,
    no_cache: bool,
    jobs: int | None,
    order: str,
    event_quota: bool,
    balance: float,
) -> None:
    """Create TITLE.html and TITLE_assets/ summarising the photos below SRC."""
    cache = None if no_cache else cache_dir or Path(platformdirs.user_cache_dir("mkgallery"))
    try:
        page = build_gallery(
            src,
            title,
            count=count,
            output_dir=output_dir,
            cache_dir=cache,
            jobs=jobs or os.cpu_count() or 1,
            order=order,
            event_quota=event_quota,
            balance=balance,
            log=lambda message: click.echo(message, err=True),
            progress=_progress,
        )
    except GalleryError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote {page}")
