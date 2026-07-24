"""
Visualization for Fisher-Rao information geometry.

Each case renders an animated GIF and a static multi-panel PNG of diagnostics.
Renderers read only the result envelope, so the same visuals can be regenerated
from a NetCDF archive by reconstructing that envelope.

Animations are drawn on a deep ink ground and printed diagnostics on warm
vellum, using the analogous-hue gradients defined in palette.py. Parameters are
labelled with their mathematical symbols throughout: mathtext in titles, axis
labels, and annotations, and the corresponding unicode letters inside the
monospace metric panels, where mathtext would break column alignment.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from tqdm import tqdm

from . import palette as P
from ..core import geometry as geo

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.linewidth'] = 0.9


def _metrics_box(ax, title, lines, dark=False):
    """Render a monospace metric panel. Greek is written as literal unicode."""
    ax.axis('off')
    fg = P.VELLUM if dark else P.GRAPHITE
    face = P.INK_SOFT if dark else '#efe9de'
    edge = '#2a3550' if dark else '#d9d0c0'
    txt = title + "\n" + "\u2500" * 36 + "\n" + "\n".join(lines)
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9.5,
            family='monospace', va='top', color=fg,
            bbox=dict(boxstyle='round,pad=0.6', facecolor=face,
                      edgecolor=edge, linewidth=1.0))


def _style_paper(fig, axes):
    """Apply the printed-panel styling to a grid of axes."""
    fig.patch.set_facecolor(P.PAPER)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(P.PAPER)
        ax.tick_params(colors=P.GRAPHITE, labelsize=9)
        for s in ax.spines.values():
            s.set_color('#c9c0b0')


def _style_3d(ax, dark=True):
    """Strip the default 3D chrome so the geometry carries the frame."""
    ground = P.INK if dark else P.PAPER
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(ground)
        axis.pane.set_alpha(1.0)
        axis.line.set_color(P.MUTED if dark else '#c9c0b0')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


class Animator:
    """GIF and PNG rendering dispatched by case kind."""

    @staticmethod
    def create_gif(result, filename, output_dir="outputs", fps=30, dpi=150,
                   colormap="jl_aurora", marker_size=8, alpha=0.85):
        kind = result['kind']
        dispatch = {
            'tessellation': Animator._gif_tessellation,
            'sphere': Animator._gif_sphere,
            'dual_weave': Animator._gif_dual_weave,
            'diffusion': Animator._gif_diffusion,
        }
        return dispatch[kind](result, filename, output_dir, fps, dpi,
                              colormap, marker_size, alpha)

    @staticmethod
    def create_diagnostics(result, filename, output_dir="outputs", dpi=150):
        kind = result['kind']
        dispatch = {
            'tessellation': Animator._diag_tessellation,
            'sphere': Animator._diag_sphere,
            'dual_weave': Animator._diag_dual_weave,
            'diffusion': Animator._diag_diffusion,
        }
        return dispatch[kind](result, filename, output_dir, dpi)

    @staticmethod
    def _prepare(filename, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out / filename

    @staticmethod
    def _save_anim(anim, filepath, fps, dpi, n_frames):
        writer = animation.PillowWriter(fps=fps)
        with tqdm(total=n_frames, desc="    Rendering", unit="frame") as pbar:
            def cb(cur, tot):
                pbar.n = cur + 1
                pbar.refresh()
            anim.save(str(filepath), writer=writer, dpi=dpi,
                      savefig_kwargs={'facecolor': P.INK},
                      progress_callback=cb)

    # ================================================================ CASE 1
    @staticmethod
    def _gif_tessellation(result, filename, output_dir, fps, dpi, colormap,
                          marker_size, alpha):
        filepath = Animator._prepare(filename, output_dir)
        edges_half = result['edges_half']
        anchors_half = result['anchors_half']
        scale = result['anchor_scale']
        frames = result['frames']
        cmap = P.get_cmap(colormap, 'jl_aurora')

        sc = (scale - scale.min()) / (np.ptp(scale) + 1e-12)
        fig, ax = plt.subplots(figsize=(9, 9), facecolor=P.INK)
        ax.set_facecolor(P.INK)
        ax.set_xlim(-1.06, 1.06)
        ax.set_ylim(-1.06, 1.06)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=P.MUTED,
                                lw=1.2, alpha=0.55))
        ax.set_title(r'Gaussian manifold as the hyperbolic plane,   $K = -1/2$',
                     color=P.VELLUM, fontsize=13.5, pad=16)
        flow_text = ax.text(0.02, 0.985, '', transform=ax.transAxes,
                            color=P.GOLD, fontsize=12.5, va='top')
        ax.text(0.02, 0.03,
                r'$\Gamma = \mathrm{PSL}(2,\mathbb{Z})$     '
                r'$(\mu,\sigma) \mapsto (\mu/\sqrt{2},\ \sigma)$',
                transform=ax.transAxes, color=P.MUTED, fontsize=10.5)

        edge_lines = [ax.plot([], [], lw=0.95,
                              color=cmap(0.18 + 0.62 * sc[e // 3]))[0]
                      for e in range(edges_half.shape[0])]
        anchor_scatter = ax.scatter([], [], s=marker_size * 3.2, zorder=5,
                                    edgecolors='none')

        def update(f):
            s = frames[f]
            for e in range(edges_half.shape[0]):
                u, v = geo.halfplane_to_disk_batch(
                    edges_half[e, :, 0] + s, edges_half[e, :, 1])
                edge_lines[e].set_data(u, v)
                rad = np.sqrt(u ** 2 + v ** 2).max()
                edge_lines[e].set_alpha(float(np.clip(1.15 - rad, 0.06, 0.8)))
            au, av = geo.halfplane_to_disk_batch(
                anchors_half[:, 0] + s, anchors_half[:, 1])
            anchor_scatter.set_offsets(np.column_stack([au, av]))
            anchor_scatter.set_color(cmap(0.25 + 0.65 * sc))
            flow_text.set_text(rf'$z \mapsto z + {s:.3f}$')
            return edge_lines + [anchor_scatter, flow_text]

        anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                       interval=1000 / fps, blit=False)
        Animator._save_anim(anim, filepath, fps, dpi, len(frames))
        plt.close(fig)
        print(f"    Animation saved: {filepath}")
        return str(filepath)

    @staticmethod
    def _diag_tessellation(result, filename, output_dir, dpi):
        filepath = Animator._prepare(filename, output_dir)
        edges = result['edges_disk']
        scale = result['anchor_scale']
        me = result['metrics']
        sc = (scale - scale.min()) / (np.ptp(scale) + 1e-12)
        cmap = P.get_cmap('jl_aurora')

        fig, axes = plt.subplots(2, 2, figsize=(14, 13))
        _style_paper(fig, axes)
        fig.suptitle(f"{result['case']}: Diagnostics", fontsize=15,
                     fontweight='bold', color=P.GRAPHITE)

        ax = axes[0, 0]
        ax.set_facecolor(P.INK)
        for e in range(edges.shape[0]):
            ax.plot(edges[e, :, 0], edges[e, :, 1], lw=0.75, alpha=0.7,
                    color=cmap(0.18 + 0.62 * sc[e // 3]))
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=P.MUTED, lw=1.2))
        ax.set_xlim(-1.06, 1.06)
        ax.set_ylim(-1.06, 1.06)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Modular tessellation of $\\mathbb{H}$ '
                     '(Poincar\u00e9 disk)',
                     fontweight='bold', color=P.GRAPHITE)

        ax = axes[0, 1]
        im = ax.pcolormesh(result['field_x'], result['field_y'], result['field'],
                           cmap=P.get_cmap('jl_ember'), shading='auto')
        ax.set_xlabel(r'$x = \mu/\sqrt{2}$', fontsize=12)
        ax.set_ylabel(r'$y = \sigma$', fontsize=12)
        ax.set_title(r'Distance to the orbit,  $\min_\gamma d(z, \gamma z_0)$',
                     fontweight='bold', color=P.GRAPHITE)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label('hyperbolic distance')

        ax = axes[1, 0]
        anchors = result['anchors_gauss']
        ax.scatter(anchors[:, 0], anchors[:, 1], c=sc, cmap=cmap, s=30,
                   edgecolors='none')
        ax.set_xlabel(r'$\mu$', fontsize=12)
        ax.set_ylabel(r'$\sigma$', fontsize=12)
        ax.set_title(r'Tile anchors in $(\mu, \sigma)$', fontweight='bold',
                     color=P.GRAPHITE)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)

        _metrics_box(axes[1, 1], "TESSELLATION METRICS", [
            f"tiles                 : {me['n_tiles']}",
            f"max |matrix entry|    : {me['max_matrix_entry']}",
            f"fundamental area      : {me['fundamental_domain_area']:.6f}",
            f"  exact \u03c0/3           : {np.pi/3:.6f}",
            f"modular relations res.: {me['modular_relations_residual']}",
            f"curvature K           : {me['curvature']:.4f}",
            f"curvature max dev.    : {me['curvature_max_deviation']:.2e}",
            f"anchor \u03c3 range        : [{me['anchor_sigma_min']:.3f},"
            f" {me['anchor_sigma_max']:.3f}]",
        ])
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor=P.PAPER)
        plt.close(fig)
        print(f"    Diagnostics saved: {filepath}")
        return str(filepath)

    # ================================================================ CASE 2
    @staticmethod
    def _gif_sphere(result, filename, output_dir, fps, dpi, colormap,
                    marker_size, alpha):
        filepath = Animator._prepare(filename, output_dir)
        arcs = result['arcs']
        tracer = result['tracer']
        frames = result['frames']
        orbit = result['orbit_points']
        orbit_ent = result['orbit_entropy']
        cmap = P.truncate(P.get_cmap(colormap, 'jl_nacre'), 0.28, 1.0)
        e_lo, e_hi = float(orbit_ent.min()), float(orbit_ent.max())

        fig = plt.figure(figsize=(9, 9), facecolor=P.INK)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(P.INK)
        for a in arcs:
            ax.plot(a[:, 0], a[:, 1], a[:, 2], color=P.TEAL, lw=0.55, alpha=0.3)
        swarm = ax.scatter(orbit[0, :, 0], orbit[0, :, 1], orbit[0, :, 2],
                           c=orbit_ent[0], cmap=cmap, vmin=e_lo, vmax=e_hi,
                           s=marker_size * 5.5, depthshade=False,
                           edgecolors='none')
        trail, = ax.plot([], [], [], color=P.GOLD, lw=2.4, alpha=0.95)
        head = ax.scatter([], [], [], color=P.VELLUM, s=62, edgecolors='none')

        _style_3d(ax, dark=True)
        ax.set_title(r'Categorical manifold as the sphere,   $K = +1/4$',
                     color=P.VELLUM, fontsize=13.5, pad=6)
        ax.text2D(0.02, 0.97,
                  r'$p \mapsto 2\sqrt{p}$,    '
                  r'$d(p,q) = 2\arccos \sum_i \sqrt{p_i q_i}$',
                  transform=ax.transAxes, color=P.MUTED, fontsize=10.5)
        ent_text = ax.text2D(0.02, 0.05, '', transform=ax.transAxes,
                             color=P.GOLD, fontsize=11.5)
        lim = 2.05
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_zlim(0, lim)

        def update(f):
            ax.view_init(elev=27, azim=np.degrees(frames[f]))
            swarm._offsets3d = (orbit[f, :, 0], orbit[f, :, 1], orbit[f, :, 2])
            swarm.set_array(orbit_ent[f])
            k = max(2, f + 1)
            seg = tracer[max(0, f - 30):k]
            trail.set_data(seg[:, 0], seg[:, 1])
            trail.set_3d_properties(seg[:, 2])
            head._offsets3d = ([tracer[f, 0]], [tracer[f, 1]], [tracer[f, 2]])
            ent_text.set_text(
                rf'$\langle H \rangle = {orbit_ent[f].mean():.3f}$ nats')
            return [swarm, trail, head, ent_text]

        anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                       interval=1000 / fps, blit=False)
        Animator._save_anim(anim, filepath, fps, dpi, len(frames))
        plt.close(fig)
        print(f"    Animation saved: {filepath}")
        return str(filepath)

    @staticmethod
    def _diag_sphere(result, filename, output_dir, dpi):
        filepath = Animator._prepare(filename, output_dir)
        arcs = result['arcs']
        centroids = result['centroids']
        cp = result['centroid_p']
        me = result['metrics']
        ent = np.array([-(p[p > 0] * np.log(p[p > 0])).sum() for p in cp])
        cmap = P.get_cmap('jl_nacre')

        fig = plt.figure(figsize=(14, 13), facecolor=P.PAPER)
        fig.suptitle(f"{result['case']}: Diagnostics", fontsize=15,
                     fontweight='bold', color=P.GRAPHITE)

        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.set_facecolor(P.PAPER)
        for a in arcs:
            ax.plot(a[:, 0], a[:, 1], a[:, 2], color='#3f8f88', lw=0.55,
                    alpha=0.6)
        sc = ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                        c=ent, cmap=cmap, s=24, edgecolors='none')
        _style_3d(ax, dark=False)
        ax.set_title(r'Octant tessellation, centroids by $H(p)$',
                     fontweight='bold', color=P.GRAPHITE)
        ax.view_init(elev=27, azim=35)
        cb = plt.colorbar(sc, ax=ax, shrink=0.6)
        cb.set_label(r'$H(p)$  [nats]')

        ax = fig.add_subplot(2, 2, 2)
        ax.set_facecolor(P.PAPER)
        im = ax.imshow(result['entropy_field'], origin='lower', cmap=cmap,
                       extent=[0, 1, 0, 1])
        ax.set_xlabel(r'$p_0$', fontsize=12)
        ax.set_ylabel(r'$p_1$', fontsize=12)
        ax.set_title(r'Shannon entropy  $H(p) = -\sum_i p_i \log p_i$',
                     fontweight='bold', color=P.GRAPHITE)
        ax.tick_params(colors=P.GRAPHITE, labelsize=9)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(r'$H(p)$  [nats]')

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.set_facecolor(P.PAPER)
        for g in result['geodesics']:
            ax.plot(g[:, 0], g[:, 1], g[:, 2], color=P.ROSE, lw=1.7)
        _style_3d(ax, dark=False)
        ax.set_title('Fisher-Rao geodesics (great-circle arcs)',
                     fontweight='bold', color=P.GRAPHITE)
        ax.view_init(elev=27, azim=35)

        ax = fig.add_subplot(2, 2, 4)
        _metrics_box(ax, "CATEGORICAL SPHERE METRICS", [
            f"triangle ineq. viol.  : {me['triangle_inequality_max_violation']:.2e}",
            f"geodesic len. residual: {me['geodesic_length_residual']:.2e}",
            f"curvature K           : {me['curvature']:.4f}",
            f"entropy H min         : {me['entropy_min']:.4f}",
            f"entropy H max         : {me['entropy_max']:.4f}",
            f"uniform ln(3)         : {me['entropy_uniform']:.4f}",
        ])
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor=P.PAPER)
        plt.close(fig)
        print(f"    Diagnostics saved: {filepath}")
        return str(filepath)

    # ================================================================ CASE 3
    @staticmethod
    def _gif_dual_weave(result, filename, output_dir, fps, dpi, colormap,
                        marker_size, alpha):
        filepath = Animator._prepare(filename, output_dir)
        e_lines = result['e_lines']
        m_lines = result['m_lines']
        triples = result['triples']
        m_legs, e_legs = result['m_legs'], result['e_legs']
        Q_path = result['Q_path']
        residuals = result['residual_series']
        frames = result['frames']
        fields = result['asymmetry_fields']
        MU, SIG = result['field_mu'], result['field_sigma']
        cmap = P.get_cmap(colormap, 'jl_duality')
        vmax = float(np.nanmax(np.abs(fields)))
        extent = [MU.min(), MU.max(), SIG.min(), SIG.max()]

        fig, ax = plt.subplots(figsize=(9.5, 8.5), facecolor=P.INK)
        ax.set_facecolor(P.INK)
        im = ax.imshow(fields[0], origin='lower', extent=extent, aspect='auto',
                       cmap=cmap, alpha=0.62, vmin=-vmax, vmax=vmax)
        for ln in e_lines:
            ax.plot(ln[:, 0], ln[:, 1], color=P.TEAL, lw=0.85, alpha=0.6)
        for ln in m_lines:
            ax.plot(ln[:, 0], ln[:, 1], color=P.GOLD, lw=0.85, alpha=0.6)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(r'$\mu$', color=P.VELLUM, fontsize=13)
        ax.set_ylabel(r'$\sigma$', color=P.VELLUM, fontsize=13)
        ax.tick_params(colors=P.MUTED)
        for s in ax.spines.values():
            s.set_color('#2a3550')
        ax.set_title(r'Dually flat weave:  $\theta$-geodesics $\perp$ '
                     r'$\eta$-geodesics',
                     color=P.VELLUM, fontsize=13, pad=14)

        ax.plot(Q_path[:, 0], Q_path[:, 1], color=P.MUTED, lw=0.9, ls=':',
                alpha=0.65)
        m_line, = ax.plot([], [], color=P.VELLUM, lw=2.5, zorder=6)
        e_line, = ax.plot([], [], color=P.JADE, lw=2.5, zorder=6)
        pts = ax.scatter(triples[0][:, 0], triples[0][:, 1], s=72, zorder=7,
                         c=[P.GOLD, P.VELLUM, P.JADE], edgecolors='none')
        labels = [ax.annotate(l, (0, 0), color=c, fontsize=14,
                              fontweight='bold', xytext=(8, 8),
                              textcoords='offset points', zorder=8)
                  for l, c in ((r'$P$', P.GOLD), (r'$Q$', P.VELLUM),
                               (r'$R$', P.JADE))]
        legend = ax.text(0.02, 0.97,
                         r'$\eta$-geodesic $P \to Q$     '
                         r'$\theta$-geodesic $Q \to R$',
                         transform=ax.transAxes, color=P.MUTED, fontsize=10.5,
                         va='top')
        readout = ax.text(0.02, 0.915, '', transform=ax.transAxes,
                          color=P.GOLD, fontsize=11.5, va='top')

        def update(f):
            im.set_data(fields[f])
            m_line.set_data(m_legs[f][:, 0], m_legs[f][:, 1])
            e_line.set_data(e_legs[f][:, 0], e_legs[f][:, 1])
            pts.set_offsets(triples[f])
            for k, lab in enumerate(labels):
                lab.set_position((triples[f, k, 0], triples[f, k, 1]))
                lab.xy = (triples[f, k, 0], triples[f, k, 1])
            readout.set_text(
                rf'$D(P\|R) - [\,D(P\|Q) + D(Q\|R)\,] = {residuals[f]:.1e}$')
            return [im, m_line, e_line, pts, readout, legend] + labels

        anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                       interval=1000 / fps, blit=False)
        Animator._save_anim(anim, filepath, fps, dpi, len(frames))
        plt.close(fig)
        print(f"    Animation saved: {filepath}")
        return str(filepath)

    @staticmethod
    def _diag_dual_weave(result, filename, output_dir, dpi):
        filepath = Animator._prepare(filename, output_dir)
        e_lines, m_lines = result['e_lines'], result['m_lines']
        me = result['metrics']
        triples = result['triples']
        frames = result['frames']
        dser = result['divergence_series']
        res_ser = result['residual_series']
        P0, Q0, R0 = triples[0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 13))
        _style_paper(fig, axes)
        fig.suptitle(f"{result['case']}: Diagnostics", fontsize=15,
                     fontweight='bold', color=P.GRAPHITE)

        ax = axes[0, 0]
        for ln in e_lines:
            ax.plot(ln[:, 0], ln[:, 1], color='#2f8f88', lw=0.85, alpha=0.85)
        for ln in m_lines:
            ax.plot(ln[:, 0], ln[:, 1], color='#c99135', lw=0.85, alpha=0.85)
        ax.plot(result['Q_path'][:, 0], result['Q_path'][:, 1],
                color=P.GRAPHITE, lw=1.2, ls=':', label=r'sweep path of $Q$')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.set_xlabel(r'$\mu$', fontsize=12)
        ax.set_ylabel(r'$\sigma$', fontsize=12)
        ax.set_title(r'Conjugate grids: $\theta$ (teal), $\eta$ (gold)',
                     fontweight='bold', color=P.GRAPHITE)

        ax = axes[0, 1]
        im = ax.pcolormesh(result['field_mu'], result['field_sigma'],
                           result['asymmetry_fields'][0],
                           cmap=P.get_cmap('jl_duality'), shading='auto')
        for pt, lab in ((P0, r'$P$'), (Q0, r'$Q$'), (R0, r'$R$')):
            ax.scatter([pt[0]], [pt[1]], color=P.GRAPHITE, s=42, zorder=5)
            ax.annotate(lab, (pt[0], pt[1]), fontsize=13, fontweight='bold',
                        xytext=(6, 6), textcoords='offset points',
                        color=P.GRAPHITE)
        ax.set_xlabel(r'$\mu$', fontsize=12)
        ax.set_ylabel(r'$\sigma$', fontsize=12)
        ax.set_title(r'Divergence asymmetry  $D(p\|Q) - D(Q\|p)$',
                     fontweight='bold', color=P.GRAPHITE)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label('nats')

        ax = axes[1, 0]
        ax.plot(frames, dser[:, 0], color='#2f8f88', lw=2, label=r'$D(P\|Q)$')
        ax.plot(frames, dser[:, 1], color='#c99135', lw=2, label=r'$D(Q\|R)$')
        ax.plot(frames, dser[:, 0] + dser[:, 1], color=P.GRAPHITE, lw=4,
                alpha=0.35, label=r'$D(P\|Q) + D(Q\|R)$')
        ax.plot(frames, dser[:, 2], color=P.ROSE, lw=1.5, ls='--',
                label=r'$D(P\|R)$')
        ax.set_xlabel('sweep parameter')
        ax.set_ylabel('divergence  [nats]')
        ax.set_title('Generalized Pythagorean identity across the sweep',
                     fontweight='bold', color=P.GRAPHITE)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2)

        ax = axes[1, 1]
        axr = fig.add_axes([0.56, 0.30, 0.38, 0.12])
        axr.set_facecolor(P.PAPER)
        axr.semilogy(frames, np.maximum(res_ser, 1e-18), color=P.VIOLET, lw=1.6)
        axr.axhline(np.finfo(float).eps, color=P.GRAPHITE, ls=':', lw=1,
                    label=r'$\varepsilon_{\mathrm{mach}}$')
        axr.set_title('residual per frame', fontsize=10, color=P.GRAPHITE)
        axr.set_xlabel('sweep parameter', fontsize=9)
        axr.legend(fontsize=8)
        axr.grid(True, alpha=0.2)
        axr.tick_params(colors=P.GRAPHITE, labelsize=8)
        _metrics_box(ax, "DUAL WEAVE METRICS", [
            f"frames verified       : {me['frames_verified']}",
            f"Pythagorean res. max  : {me['pythagorean_residual']:.2e}",
            f"Pythagorean res. mean : {me['pythagorean_residual_mean']:.2e}",
            f"orthogonality residual: {me['orthogonality_residual']:.2e}",
            f"D(P\u2016Q) frame 0        : {me['D_PQ']:.6f}",
            f"D(Q\u2016R) frame 0        : {me['D_QR']:.6f}",
            f"D(P\u2016R) frame 0        : {me['D_PR']:.6f}",
            f"KL asymmetry max      : {me['kl_asymmetry_max']:.4f}",
            f"curvature K           : {me['curvature']:.4f}",
        ])
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor=P.PAPER)
        plt.close(fig)
        print(f"    Diagnostics saved: {filepath}")
        return str(filepath)

    # ================================================================ CASE 4
    @staticmethod
    def _gif_diffusion(result, filename, output_dir, fps, dpi, colormap,
                       marker_size, alpha):
        filepath = Animator._prepare(filename, output_dir)
        U, V = result['disk_u'], result['disk_v']
        dist = result['dist_from_start']
        ref = result['ref_geodesic_disk']
        frames = result['frames']
        # Marks sit on the ink ground, where the darkest fifth of the ramp is
        # indistinguishable from the background. Trimming it costs hue range
        # that was never visible here and lifts the low end to a legible
        # contrast. The printed diagnostics keep the full ramp, which has
        # excellent contrast on vellum.
        cmap = P.truncate(P.get_cmap(colormap, 'jl_ember'), 0.38, 1.0)
        vmax = float(np.percentile(dist, 98))

        fig, ax = plt.subplots(figsize=(9, 9), facecolor=P.INK)
        ax.set_facecolor(P.INK)
        ax.set_xlim(-1.06, 1.06)
        ax.set_ylim(-1.06, 1.06)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=P.MUTED,
                                lw=1.2, alpha=0.55))
        ax.plot(ref[:, 0], ref[:, 1], color=P.MUTED, lw=1.0, ls='--', alpha=0.45)
        ax.set_title(r'Brownian motion of Gaussians,   $K = -1/2$',
                     color=P.VELLUM, fontsize=13.5, pad=16)
        ax.text(0.02, 0.03,
                r'$dx = \sigma\, dW_1$,     '
                r'$d(\log \sigma) = dW_2 - \frac{1}{2} dt$',
                transform=ax.transAxes, color=P.MUTED, fontsize=10.5)
        scatter = ax.scatter(U[0], V[0], s=marker_size, c=dist[0],
                             cmap=cmap, vmin=0, vmax=vmax, alpha=alpha,
                             edgecolors='none')
        time_text = ax.text(0.02, 0.985, '', transform=ax.transAxes,
                            color=P.GOLD, fontsize=12.5, va='top')

        def update(f):
            scatter.set_offsets(np.column_stack([U[f], V[f]]))
            scatter.set_array(dist[f])
            time_text.set_text(
                rf'$t = {frames[f]:.3f}$      '
                rf'$\langle d_{{FR}} \rangle = {dist[f].mean():.3f}$')
            return [scatter, time_text]

        anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                       interval=1000 / fps, blit=False)
        Animator._save_anim(anim, filepath, fps, dpi, len(frames))
        plt.close(fig)
        print(f"    Animation saved: {filepath}")
        return str(filepath)

    @staticmethod
    def _diag_diffusion(result, filename, output_dir, dpi):
        filepath = Animator._prepare(filename, output_dir)
        times = result['frames']
        sigma = result['sigma']
        dist = result['dist_from_start']
        ent = result['entropy_series']
        me = result['metrics']
        cmap = P.get_cmap('jl_ember')

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        _style_paper(fig, axes)
        fig.suptitle(f"{result['case']}: Diagnostics", fontsize=15,
                     fontweight='bold', color=P.GRAPHITE)

        ax = axes[0, 0]
        ax.set_facecolor(P.INK)
        sc = ax.scatter(result['disk_u'][-1], result['disk_v'][-1],
                        c=dist[-1], cmap=cmap, s=11, edgecolors='none')
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=P.MUTED, lw=1.2))
        ax.set_xlim(-1.06, 1.06)
        ax.set_ylim(-1.06, 1.06)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Final walker cloud (Poincare disk)', fontweight='bold',
                     color=P.GRAPHITE)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(r'$d_{FR}$ from origin')

        ax = axes[0, 1]
        mean_dist = dist.mean(axis=1)
        ax.plot(times, mean_dist, color=P.ROSE, lw=2,
                label=r'$\langle d_{FR} \rangle$')
        ax.plot(times, me['escape_rate'] * times + me['escape_intercept'],
                color=P.GRAPHITE, ls='--', lw=1.2,
                label=rf"slope $= {me['escape_rate']:.3f}$")
        ax.fill_between(times, np.percentile(dist, 10, axis=1),
                        np.percentile(dist, 90, axis=1), color=P.ROSE,
                        alpha=0.15)
        ax.set_xlabel(r'diffusion time  $t$')
        ax.set_ylabel(r'$d_{FR}$')
        ax.set_title('Boundary escape: linear growth of distance',
                     fontweight='bold', color=P.GRAPHITE)
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.2)

        ax = axes[1, 0]
        ax.plot(times, sigma.mean(axis=1), color='#2f8f88', lw=2,
                label=r'$\mathbb{E}[\sigma]$   (martingale)')
        ax.axhline(me['mean_sigma_initial_ref'], color='#2f8f88', ls=':',
                   alpha=0.7)
        ax.plot(times, np.exp(np.log(sigma).mean(axis=1)), color='#c99135',
                lw=2, label=r'$\exp \mathbb{E}[\log \sigma]$   (drifts down)')
        ax.set_xlabel(r'diffusion time  $t$')
        ax.set_ylabel(r'$\sigma$')
        ax.set_title(r'$\mathbb{E}[\sigma]$ conserved,  '
                     r'$\mathbb{E}[\log \sigma]$ drifts down',
                     fontweight='bold', color=P.GRAPHITE)
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.2)

        ax = axes[1, 1]
        axt = fig.add_axes([0.56, 0.30, 0.38, 0.13])
        axt.set_facecolor(P.PAPER)
        axt.plot(times, ent, color=P.VIOLET, lw=2)
        axt.set_title(r'ensemble entropy  $H(t)$', fontsize=10,
                      color=P.GRAPHITE)
        axt.set_xlabel(r'$t$', fontsize=9)
        axt.grid(True, alpha=0.2)
        axt.tick_params(colors=P.GRAPHITE, labelsize=8)
        _metrics_box(ax, "DIFFUSION METRICS", [
            f"escape rate           : {me['escape_rate']:.4f}",
            f"E[\u03c3] final            : {me['mean_sigma_final']:.4f}",
            f"martingale deviation  : {me['martingale_deviation']:.2e}",
            f"E[log \u03c3] slope        : {me['log_sigma_slope']:.4f}",
            f"entropy growth \u0394H     : {me['entropy_growth']:.4f}",
            f"curvature K           : {me['curvature']:.4f}",
        ])
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor=P.PAPER)
        plt.close(fig)
        print(f"    Diagnostics saved: {filepath}")
        return str(filepath)
