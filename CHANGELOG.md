# Changelog

## [Unreleased] - 2024-feb-15
* Migrate to poetry to better handle package/dev dependencies
    * Include `black` and `ruff` for formatting and linting
    * Include `mypy` for check proper typing
* Added explicit exception `DCM2NIIXError` in case of requiring future handling
* Add `.gitignore`
* Add `Makefile` for local and CI testing
* Remove print and add proper logger to better identify the origing of the logs
* Refacto aiming to have:
    * More explicit imports
    * Explicit typying
* Remove `pacsman_data/data` from package (non required)
* Fix wrong class instantiation during synthetic data generation [`pacsman_data/generate_dummy_images.py#L138`](https://github.com/sssilvar/PACSMAN_data/blame/e0194e98c6731d395f58edc6bc358904baf1194b/pacsman_data/generate_dummy_images.py#L138)
