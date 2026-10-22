"""Faults that belong to the service, not to the caller."""

from __future__ import annotations


class ServiceFault(RuntimeError):
    """Something on the server side is wrong. The request was not at fault."""

    status_code = 500

    public_detail = (
        "The prediction service could not complete this request because of a "
        "server-side fault. The request itself was not rejected. Check the "
        "server logs."
    )

    def __init__(self, message: str, public_detail: str | None = None) -> None:
        super().__init__(message)
        if public_detail is not None:
            self.public_detail = public_detail


class ArtifactUnavailable(ServiceFault):
    """A model bundle, calibration or export the service needs is not on disk.

    Reported as 503 rather than 500 because it is a deployment state that a
    rebuild or re-export fixes, not a defect in the running code.
    """

    status_code = 503

    public_detail = (
        "A model artefact this endpoint depends on is not available on the "
        "server. The request was not rejected. Rebuild or re-export the "
        "artefacts and retry."
    )


def predict_probabilities(model, frame, stream: str):
    """Call ``predict_proba`` and classify anything it raises as our fault.

    By the time a frame reaches the model it has already been through the
    stream's own input validation, so a failure here is a mismatch between the
    frozen pipeline and the frame it was handed - a bundle problem, not a
    request problem. Letting the raw ``ValueError`` escape made it
    indistinguishable from "you left out training_load", which is the confusion
    this whole module exists to remove.
    """
    if model is None:
        raise ServiceFault(f"The {stream} bundle carries no usable serving model.")
    try:
        return model.predict_proba(frame)
    except ServiceFault:
        raise
    except Exception as error:
        raise ServiceFault(
            f"The {stream} model failed while scoring a validated feature row: "
            f"{type(error).__name__}: {error}"
        ) from error


__all__ = ["ArtifactUnavailable", "ServiceFault", "predict_probabilities"]
