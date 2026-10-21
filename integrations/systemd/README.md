# systemd (optional, Linux)

`ldr run` needs no service manager: it is resumable, so after a crash or Ctrl-C the same command continues. This
template only makes a long run start again by itself after a crash or reboot.

1. Fill in the placeholders and install the unit:

   ```sh
   sed -e "s|@LDR@|$(command -v ldr)|" \
       -e "s|@ENV_FILE@|$HOME/.config/local-deep-research/env|" \
       -e "s|@CACHE_DIR@|$HOME/.cache/local-deep-research|" \
       -e "s|@MEMORY_MAX@|20G|" \
       ldr-research@.service > ~/.config/systemd/user/ldr-research@.service
   systemctl --user daemon-reload
   ```

   | placeholder | meaning |
   |---|---|
   | `@LDR@` | absolute path of the `ldr` command (from the venv you installed it into) |
   | `@ENV_FILE@` | optional file with `LDR_*` settings, e.g. `LDR_LLM_BASE_URL=http://localhost:8000/v1` and `LDR_LLM_API_KEY=...` (the `-` prefix in the unit makes it optional) |
   | `@CACHE_DIR@` | the page cache (`cache_dir` setting; default `~/.cache/local-deep-research`) |
   | `@MEMORY_MAX@` | memory cap for the run, e.g. `20G` |

2. Create the topic and start it (the instance name is the escaped topic path):

   ```sh
   ldr new "your question"                      # prints the topic folder
   systemctl --user enable --now "ldr-research@$(systemd-escape --path /path/to/topic).service"
   loginctl enable-linger "$USER"               # keep user services running after logout / start them at boot
   ```

3. Follow it: `ldr status /path/to/topic` or `journalctl --user -u "ldr-research@*" -f`.

Options of `ldr run` (profile, languages, output language) are stored in `research/harvest/profile.json` by the first
run; set them once by hand (`ldr run /path/to/topic --profile derin --until profile`) before enabling the service.
If your LLM server or SearXNG are user services too, add them to `After=` / `Wants=` in the unit.
