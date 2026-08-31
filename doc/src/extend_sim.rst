.. index:: extend_sim

extend_sim command
==================

Syntax
""""""

.. code-block:: LAMMPS

   extend_sim init file nreplica N keyword value ...

* init file = LAMMPS data file holding one pristine copy of the sample to append
* nreplica N = number of copies of the sample to stack onto the end (N <= 0 is a no-op)
* zero or more keyword/value pairs may be appended
* keyword = *restart* or *axis* or *new_end* or *gap* or *frozen_depth*

  .. parsed-literal::

       *restart* value = file
         file = restart file to load as the base system before appending
       *axis* value = *x* or *y* or *z*
         growth direction (default = *z*)
       *new_end* value = coord
         final box coordinate on the high side of *axis* (box units);
         default = current high edge + gap + N * (sample thickness)
       *gap* value = d
         empty spacing (box units) inserted below the first appended
         copy (default 0)
       *frozen_depth* value = d
         reserved slab thickness (box units); currently only range-checked

Examples
""""""""

For runnable input scripts (append to a live system, extend from a
restart, and a piston shock with on-the-fly extension) see
examples/extend_sim.

.. code-block:: LAMMPS

   extend_sim init sample.data nreplica 3
   extend_sim init sample.data nreplica 2 axis z new_end 180.0
   extend_sim init sample.data nreplica 1 axis z gap 1.7
   extend_sim restart base.restart init sample.data axis z nreplica 4

Description
"""""""""""

Append one or more copies of a reference sample to the high end of the
simulation box along one axis.  The sample is read from a LAMMPS
:doc:`data file <read_data>` (*init*); its thickness ``L`` along *axis*
is taken from the ``xlo xhi`` / ``ylo yhi`` / ``zlo zhi`` line of that
file's header.  Copy ``i`` (``i = 0 .. N-1``) is placed so that its
lower face sits at ``hi_old + gap + i*L``, where ``hi_old`` is the
current high edge of the box along *axis*.  With ``gap = 0`` (the
default) the copies tile the region ``[hi_old, hi_old + N*L]``
seamlessly, provided the sample is itself periodic along *axis* with
period ``L``.

When the existing system has a real surface layer *at* ``hi_old`` (any
non-periodic / shrink-wrapped growth face), stacking at ``gap = 0``
places the new copy's first layer on top of that surface layer and the
atoms overlap.  Set *gap* to roughly one interlayer spacing in that
case; the vacuum closes up once dynamics resume.

The command can obtain the base system in two ways:

* **From a restart.**  If the *restart* keyword is given, that file is
  loaded with :doc:`read_restart <read_restart>` and no simulation box
  may exist yet.  Use :doc:`clear <clear>` first if one does.
* **From the current system.**  If *restart* is omitted, the command
  operates on the system already in memory (from a previous
  :doc:`read_data <read_data>`, :doc:`read_restart <read_restart>`,
  :doc:`create_box <create_box>`, or a completed :doc:`run <run>`).  A
  box must already exist in this case.

How the box is grown depends on the boundary style of the high *axis*
face (see :doc:`boundary <boundary>`):

* *periodic* (*p*) or *fixed* (*f*): the box is enlarged with
  :doc:`change_box <change_box>` to *new_end* before the atoms are read
  in.
* *shrink-wrapped* (*s* or *m*): no ``change_box`` is issued; the box
  re-expands automatically to enclose the new atoms inside
  :doc:`read_data add <read_data>`.  *new_end* is ignored in this case.

Each copy is inserted with ``read_data init add append shift ...``, so
new atoms receive fresh atom IDs appended after the current maximum and
all per-atom attributes stored in *init* (including velocities) are
carried over.

``extend_sim`` performs only the geometry construction.  Defining a
piston, freezing a substrate slab, or thermostatting the appended
material is left to the input script with the usual
:doc:`region <region>`, :doc:`group <group>`, :doc:`velocity
<velocity>`, and :doc:`fix setforce <fix_setforce>` commands.  The
*frozen_depth* keyword is accepted for forward compatibility but is
currently only checked to be smaller than the sample thickness.

.. note::

   The sample data file must be periodic along *axis* with period equal
   to its box length along that axis, or the tiled copies will overlap
   or leave gaps.  A cell built with :doc:`lattice <lattice>` /
   :doc:`create_atoms <create_atoms>` and written with
   ``write_data ... nocoeff`` satisfies this by construction.

Restrictions
""""""""""""

This command is part of the EXTRA-COMMAND package.  It is only enabled
if LAMMPS was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.

The sample is inserted verbatim; the command does not check for atom
overlaps between the appended copies and pre-existing atoms near
``hi_old``.  With the *restart* keyword the base system's atoms may
carry nonzero image flags along a periodic growth axis, which produces
a warning from :doc:`read_data <read_data>`; it is harmless when the
growth axis is non-periodic.

Related commands
""""""""""""""""

:doc:`replicate <replicate>`, :doc:`read_data <read_data>`,
:doc:`read_restart <read_restart>`, :doc:`change_box <change_box>`,
:doc:`fix wall/piston <fix_wall_piston>`

Default
"""""""

The defaults are axis = z, gap = 0, new_end = hi_old + N*L,
frozen_depth = 0.
