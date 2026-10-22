"""The categorical values a frozen bundle's pipeline was actually fitted on."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.serving.errors import ServiceFault


def _encoder_categories(step: Any) -> Sequence[Sequence[Any]] | None:
    """The fitted categories of an encoder, or None if this is not one."""
    categories = getattr(step, "categories_", None)
    if categories is None:
        return None
    return list(categories)


def _categories_from_transformer(transformer: Any) -> Sequence[Sequence[Any]] | None:
    """Find the fitted encoder inside a column transformer's branch.

    The branch is a Pipeline of imputer then encoder in the frozen bundles, but
    a bare encoder is accepted too so this does not depend on the branch having
    exactly the shape it happens to have today.
    """
    direct = _encoder_categories(transformer)
    if direct is not None:
        return direct

    steps = getattr(transformer, "steps", None)
    if not steps:
        return None
    for _, step in reversed(list(steps)):
        categories = _encoder_categories(step)
        if categories is not None:
            return categories
    return None


def trained_categorical_vocabulary(bundle: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Map each categorical feature to the values the pipeline was fitted on.

    Values are returned as strings because that is how they are compared: the
    injury bundle encodes ``playing_surface`` as integer codes while the schema
    endpoint offers them as strings, and a comparison that distinguished ``2``
    from ``"2"`` would flag every request as out of distribution.

    Raises :class:`ServiceFault` when the vocabulary cannot be derived. Silence
    is the wrong failure here: the caller uses this to decide whether to warn
    that a request is an extrapolation, and a warning that quietly stops being
    emitted is worse than one that never existed, because the interface goes on
    looking as though it is still checking.
    """
    declared = bundle.get("categorical_options")
    if declared:
        return {
            str(feature): tuple(str(value) for value in values)
            for feature, values in dict(declared).items()
        }

    features = [str(name) for name in bundle.get("categorical_features", [])]
    if not features:
        return {}

    model = bundle.get("serving_model", bundle.get("calibrated_model"))
    if model is None:
        raise ServiceFault(
            "The bundle has categorical features but no serving model to read "
            "their fitted vocabulary from."
        )

    # The preprocessor is the first step of the serving pipeline in the frozen
    # bundles. It is located by looking for the transformers rather than by
    # name, so a renamed step does not silently disable the check.
    preprocessor = None
    for _, step in getattr(model, "steps", []) or []:
        if hasattr(step, "transformers_"):
            preprocessor = step
            break
    if preprocessor is None and hasattr(model, "transformers_"):
        preprocessor = model
    if preprocessor is None:
        raise ServiceFault(
            "No fitted ColumnTransformer was found in the serving pipeline, so "
            "the trained categorical vocabulary could not be read from it."
        )

    wanted = list(features)
    for _name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or transformer is None:
            continue
        branch_columns = [str(column) for column in (columns or [])]
        if branch_columns != wanted:
            continue

        categories = _categories_from_transformer(transformer)
        if categories is None:
            raise ServiceFault(
                "The categorical branch of the serving pipeline exposes no "
                "fitted encoder, so its vocabulary could not be read."
            )
        if len(categories) != len(wanted):
            # Positional mapping is only correct when the counts agree; a
            # mismatch would attach one feature's vocabulary to another and
            # produce confident, wrong out-of-distribution warnings.
            raise ServiceFault(
                f"The fitted encoder reports {len(categories)} category sets "
                f"for {len(wanted)} categorical features, so they cannot be "
                "mapped onto each other."
            )
        return {
            feature: tuple(str(value) for value in values)
            for feature, values in zip(wanted, categories)
        }

    raise ServiceFault(
        "No branch of the serving pipeline's ColumnTransformer covers exactly "
        f"the declared categorical features {wanted!r}, so their fitted "
        "vocabulary could not be read."
    )
