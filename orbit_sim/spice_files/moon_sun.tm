@'
KPL/MK

\begindata

KERNELS_TO_LOAD = (
   'naif0012.tls'
   'de440s.bsp'
)

\enddata
'@ | Set-Content -Path .\spice_files\moon_sun.tm -Encoding ascii
