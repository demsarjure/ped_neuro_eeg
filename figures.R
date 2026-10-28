# Figures 5-8 (run from the repository root)

# libraries and utils ----------------------------------------------------------
library(tidyverse)
library(cowplot)
library(ggdist)

source("./utils/fits.R")

args <- commandArgs(trailingOnly = TRUE)
OUT <- if (length(args) > 0) args[1] else "./figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# fonts ------------------------------------------------------------------------
# PLOS ONE accepts only Arial, Times or Symbol. Arial is used if it is installed or, under WSL,
# found among the Windows fonts; otherwise Liberation Sans (fonts/, SIL Open Font License),
# which has the same letter widths, stands in for it
if (!"Arial" %in% systemfonts::system_fonts()$family) {
  windows <- file.path("/mnt/c/Windows/Fonts", c("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"))
  if (all(file.exists(windows))) {
    systemfonts::register_font("Arial", windows[1], windows[2], windows[3], windows[4])
  } else {
    message("Arial not found, so the figures use Liberation Sans")
    systemfonts::register_font("Arial", "./fonts/LiberationSans-Regular.ttf")
  }
}
set_null_device("agg")

# style of the original figures, with all text 8-10 pt (PLOS ONE: 8-12 pt)
theme_original <- function() {
  theme_minimal(base_size = 9, base_family = "Arial") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 10),
      axis.text = element_text(size = 8),
      plot.background = element_rect(fill = "white", color = NA)
    )
}

# TIFF, RGB, LZW, 600 dpi (PLOS ONE)
save_figure <- function(plot, name, width, height) {
  ggsave(
    file.path(OUT, paste0(name, ".tif")), plot, width = width, height = height, units = "in",
    dpi = 600, device = ragg::agg_tiff, compression = "lzw", bg = "white"
  )
}

# data -------------------------------------------------------------------------
alpha <- load_metrics("alpha")
young <- alpha %>% filter(age >= 2, age <= 9)

# Figure 5: global efficiency, posterior of the mean difference (Model 1) ------
model1 <- compile_model("diff_t")
ge1 <- fit_model1(model1, alpha, "ge", "exploratory", "alpha")
mu <- as.numeric(posterior::as_draws_df(ge1$fit$draws("mu"))$mu)
cat("Figure 5: mu", round(mean(mu), 2), "P(mu < 0)", mean(mu < 0), "\n")

fig5 <- ggplot(tibble(mu = mu), aes(x = mu)) +
  stat_halfeye(fill = "skyblue", alpha = 0.75) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  xlab("Mean difference (control SD)") +
  ylab("") +
  ggtitle("Global efficiency") +
  theme_original() +
  theme(axis.text.y = element_blank())
save_figure(fig5, "Fig5", width = 5, height = 3)

# Figures 6 and 7: change with age (Model 2), 50% and 90% intervals ------------
model2 <- compile_model("group_age_t")

age_panels <- function(metric, ylab) {
  fit <- fit_model2(model2, young, metric, "figure", "alpha")
  print(fit$summary %>% select(parameter, mean, lower, upper, p_below))
  draws <- model2_draws(fit$fit)
  prep <- fit$prep

  lines <- expand_grid(group = c("test", "control"), age = seq(2, 9, by = 0.1)) %>%
    pmap_dfr(function(group, age) {
      g <- as.numeric(group == "test")
      a <- (age - 5) / (2 * prep$age_sd)
      eta <- draws$alpha + draws$b_group * g + draws$b_age * a + draws$b_group_age * g * a
      tibble(group = group, age = age, y = eta * prep$y_sd + prep$y_mean) %>%
        group_by(group, age) %>%
        mean_qi(y, .width = c(0.5, 0.9)) %>%
        ungroup()
    })

  y_range <- range(c(young[[metric]], lines$.lower, lines$.upper))
  panel <- function(g, title) {
    ggplot(lines %>% filter(group == g), aes(x = age, y = y, ymin = .lower, ymax = .upper)) +
      geom_lineribbon(linewidth = 0.5, show.legend = FALSE) +
      geom_point(
        data = young %>% filter(group == g), aes(x = age, y = .data[[metric]]),
        inherit.aes = FALSE, color = "black", alpha = 0.3, shape = 16, size = 1
      ) +
      scale_fill_brewer() +
      scale_x_continuous(breaks = 2:9) +
      coord_cartesian(ylim = y_range) +
      xlab("Age (years)") +
      ylab(ylab) +
      ggtitle(title) +
      theme_original()
  }
  plot_grid(panel("test", "Test group"), panel("control", "Control group"), ncol = 2)
}

fig6 <- age_panels("lh_rh", "Interhemispheric strength (IHS)")
save_figure(fig6, "Fig6", width = 7.5, height = 3.2)
fig7 <- age_panels("ge", "Global efficiency (GE)")
save_figure(fig7, "Fig7", width = 7.5, height = 3.2)

# Figure 8: connectivity between the hemispheres, the same pair as originally ---
ELECTRODES <- c(
  "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
  "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
)
LEFT <- c("Fp1", "F7", "F3", "T3", "C3", "T5", "P3", "O1")
RIGHT <- c("Fp2", "F8", "F4", "T4", "C4", "T6", "P4", "O2")

read_lr <- function(group, id) {
  c <- read.table(
    paste0("./data/connectomes/", group, "/alpha/", id, "_alpha_task-rest_connectome_eeg.csv"),
    header = FALSE, sep = ","
  )
  colnames(c) <- ELECTRODES
  rownames(c) <- ELECTRODES
  c %>%
    rownames_to_column("Var1") %>%
    pivot_longer(-Var1, names_to = "Var2") %>%
    filter(Var1 %in% LEFT, Var2 %in% RIGHT) %>%
    mutate(
      Var1 = factor(Var1, levels = rev(LEFT)),
      Var2 = factor(Var2, levels = RIGHT)
    )
}

pair <- bind_rows(
  read_lr("test", "T003") %>% mutate(panel = "Test participant"),
  read_lr("control", "C028") %>% mutate(panel = "Control participant")
) %>%
  mutate(panel = factor(panel, levels = c("Test participant", "Control participant")))

fig8 <- ggplot(pair, aes(Var2, Var1)) +
  geom_tile(aes(fill = value)) +
  scale_fill_gradient2(low = "white", high = "darkblue", name = "dbWPLI") +
  facet_wrap(~panel) +
  xlab("Right hemisphere electrodes") +
  ylab("Left hemisphere electrodes") +
  theme_original() +
  theme(
    axis.ticks = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    panel.grid = element_blank(),
    strip.text = element_text(size = 10),
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 8)
  )
save_figure(fig8, "Fig8", width = 6.5, height = 3.4)

cat("done\n")
