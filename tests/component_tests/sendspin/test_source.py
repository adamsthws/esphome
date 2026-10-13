"""Validation and code generation tests for the sendspin source role.

These cover the rejection branches of the source block and the generated role
config, which a compile test cannot check: a `test*.yaml` can only assert that a
configuration is accepted.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from esphome import config_validation as cv
from esphome.components.audio import CONF_MAX_SAMPLE_RATE, CONF_MIN_SAMPLE_RATE
from esphome.components.esp32.const import KEY_ESP32, KEY_SDKCONFIG_OPTIONS
from esphome.components.sendspin import (
    CONFIG_SCHEMA,
    FINAL_VALIDATE_SCHEMA,
    _get_data,
    _request_source,
)
from esphome.config import Config
from esphome.const import CONF_ID, CONF_SOURCE, PlatformFramework
from esphome.core import CORE, ID
from esphome.types import ConfigType
from tests.component_tests.types import SetCoreConfigCallable

HUB_ID = "sendspin_hub"
MIC_ID = "mic"


def _full_config(min_sample_rate: int = 16000, max_sample_rate: int = 16000) -> Config:
    """A full config carrying one microphone, as the ID pass leaves it."""
    mic: ConfigType = {
        CONF_ID: ID(MIC_ID, is_declaration=True, type="microphone"),
        CONF_MIN_SAMPLE_RATE: min_sample_rate,
        CONF_MAX_SAMPLE_RATE: max_sample_rate,
        "min_channels": 1,
        "max_channels": 2,
        "min_bits_per_sample": 16,
        "max_bits_per_sample": 32,
    }
    full = Config()
    full["microphone"] = [mic]
    full.declare_ids.append((mic[CONF_ID], ["microphone", 0, CONF_ID]))
    return full


def _hub_config(**source: Any) -> ConfigType:
    """Validate a hub config with a source block, allowing field overrides."""
    source_config: ConfigType = {
        "microphone": {"microphone": ID(MIC_ID, is_declaration=False)}
    }
    source_config.update(source)
    return CONFIG_SCHEMA({CONF_ID: HUB_ID, CONF_SOURCE: source_config})


@pytest.fixture
def microphone(set_core_config: SetCoreConfigCallable) -> None:
    set_core_config(PlatformFramework.ESP32_IDF, full_config=_full_config())


@pytest.mark.usefixtures("microphone")
def test_defaults() -> None:
    config = _hub_config()

    source = config[CONF_SOURCE]
    assert source["codec"] == "pcm"
    assert source["chunk_duration"].total_milliseconds == 20
    assert source["buffer_duration"].total_milliseconds == 500


@pytest.mark.usefixtures("microphone")
def test_final_validate_records_sample_rate() -> None:
    """The stream takes the microphone's own rate, since nothing can resample it."""
    FINAL_VALIDATE_SCHEMA(_hub_config())

    assert _get_data().source_sample_rate == 16000


@pytest.mark.usefixtures("microphone")
def test_unsupported_bits_per_sample_rejected() -> None:
    with pytest.raises(cv.Invalid, match="bits_per_sample must be one of"):
        _hub_config(microphone={"microphone": MIC_ID, "bits_per_sample": 20})


@pytest.mark.usefixtures("microphone")
def test_opus_requires_16_bits() -> None:
    with pytest.raises(cv.Invalid, match="requires a bits_per_sample of 16"):
        _hub_config(
            codec="opus", microphone={"microphone": MIC_ID, "bits_per_sample": 24}
        )


@pytest.mark.usefixtures("microphone")
def test_opus_chunk_duration_rejected() -> None:
    """Opus encodes each chunk as one frame, so only its frame sizes work."""
    with pytest.raises(cv.Invalid, match="requires chunk_duration to be one of"):
        _hub_config(codec="opus", chunk_duration="15ms")


@pytest.mark.usefixtures("microphone")
def test_short_buffer_rejected() -> None:
    """A buffer too short for one microphone read would refuse every write."""
    with pytest.raises(cv.Invalid, match="must be at least"):
        _hub_config(buffer_duration="50ms")


def test_variable_sample_rate_rejected(set_core_config: SetCoreConfigCallable) -> None:
    set_core_config(
        PlatformFramework.ESP32_IDF,
        full_config=_full_config(min_sample_rate=16000, max_sample_rate=48000),
    )

    with pytest.raises(cv.Invalid, match="must have a fixed sample rate"):
        FINAL_VALIDATE_SCHEMA(_hub_config())


def test_opus_sample_rate_rejected(set_core_config: SetCoreConfigCallable) -> None:
    set_core_config(
        PlatformFramework.ESP32_IDF,
        full_config=_full_config(min_sample_rate=44100, max_sample_rate=44100),
    )

    with pytest.raises(
        cv.Invalid, match="requires the microphone sample_rate to be one of"
    ):
        FINAL_VALIDATE_SCHEMA(_hub_config(codec="opus"))


def test_source_automation_without_source_rejected(
    set_core_config: SetCoreConfigCallable,
) -> None:
    """The source actions and condition have nothing to act on without a source block."""
    set_core_config(PlatformFramework.ESP32_IDF)
    _request_source({})

    with pytest.raises(cv.Invalid, match="require a 'source' block"):
        FINAL_VALIDATE_SCHEMA(CONFIG_SCHEMA({CONF_ID: HUB_ID}))


def _sdkconfig_option(name: str) -> Any:
    return CORE.data[KEY_ESP32][KEY_SDKCONFIG_OPTIONS].get(name)


def test_opus_source_config_generated(
    generate_main: Callable[[str | Path], str],
    component_config_path: Callable[[str], Path],
) -> None:
    """The set_signal action turns on line sense, and an Opus source keeps Opus in the library."""
    main_cpp = generate_main(component_config_path("source_opus.yaml"))

    assert "set_source_config(" in main_cpp
    assert ".sample_rate = 48000" in main_cpp
    assert ".capture_buffer_ms = 500" in main_cpp
    assert ".buffer_location = sendspin::MemoryLocation::PREFER_INTERNAL" in main_cpp
    assert ".codec = sendspin::SendspinCodecFormat::OPUS" in main_cpp
    assert ".line_sense = true" in main_cpp
    assert _sdkconfig_option("CONFIG_SENDSPIN_ENABLE_SOURCE") is None
    assert _sdkconfig_option("CONFIG_SENDSPIN_ENABLE_OPUS") is None


def test_pcm_source_config_generated(
    generate_main: Callable[[str | Path], str],
    component_config_path: Callable[[str], Path],
) -> None:
    """Without the set_signal action or buffer_memory, neither is set, and unused Opus is left out."""
    main_cpp = generate_main(component_config_path("source_pcm.yaml"))

    assert ".sample_rate = 16000" in main_cpp
    assert ".codec = sendspin::SendspinCodecFormat::PCM" in main_cpp
    assert ".line_sense = false" in main_cpp
    assert ".buffer_location" not in main_cpp
    assert _sdkconfig_option("CONFIG_SENDSPIN_ENABLE_OPUS") is False


def test_no_source_disables_the_role(
    generate_main: Callable[[str | Path], str],
    component_config_path: Callable[[str], Path],
) -> None:
    main_cpp = generate_main(component_config_path("device_info_default.yaml"))

    assert "set_source_config(" not in main_cpp
    assert _sdkconfig_option("CONFIG_SENDSPIN_ENABLE_SOURCE") is False
